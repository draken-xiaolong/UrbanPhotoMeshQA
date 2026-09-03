#!/usr/bin/env python3
"""Build full-reference Patch texture targets; deployment remains no-reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, sobel

from build_texture_targets_v2 import resolve_path, sample_texture_colors, stable_seed
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.texture import STANDARD_DIRECTIONS, render_textured_view_with_masks


ATTACKS = {"texture_detail_loss", "texture_region_missing", "texture_misalignment"}
METRICS = ("masked_rgb_rmse", "masked_gradient_mae", "masked_one_minus_ssim",
           "masked_color_mean_shift", "surface_rgb_rmse",
           "surface_changed_fraction_1_255")
LEVEL_ORDER = {"light": 0, "medium": 1, "heavy": 2}


def render(path: Path, size: int):
    mesh = GltfReader(path).load_mesh(include_texture=True)
    views = [render_textured_view_with_masks(
        mesh, direction=direction, size=size, return_face_ids=True)
        for direction in STANDARD_DIRECTIONS]
    return mesh, *(np.stack([item[index] for item in views]) for index in range(4))


def sample_patch_uv(mesh, face_patch, samples_per_patch: int, seed: int):
    triangles = np.asarray(mesh.vertices, np.float64)[mesh.faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    paths = mesh.metadata.get("material_texture_paths", [])
    materials = np.asarray(mesh.face_materials, np.int64)
    valid = (materials >= 0) & (materials < len(paths))
    valid &= np.asarray([paths[value] is not None if 0 <= value < len(paths) else False
                         for value in materials])
    valid &= np.isfinite(np.asarray(mesh.texcoords)[mesh.faces]).all(axis=(1, 2))
    rng = np.random.default_rng(seed); selected = []; uv = []; labels = []
    for patch in range(int(face_patch.max()) + 1):
        candidates = np.flatnonzero((face_patch == patch) & valid & (areas > 0))
        if not len(candidates):
            continue
        weights = areas[candidates] / areas[candidates].sum()
        faces = rng.choice(candidates, samples_per_patch, replace=True, p=weights)
        r1 = np.sqrt(rng.random((samples_per_patch, 1))); r2 = rng.random((samples_per_patch, 1))
        bary = np.concatenate([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1)
        face_uv = np.asarray(mesh.texcoords, np.float64)[mesh.faces[faces]]
        selected.append(faces); uv.append(np.sum(face_uv * bary[:, :, None], axis=1))
        labels.append(np.full(samples_per_patch, patch, np.int64))
    if not selected:
        raise ValueError("No patch has a valid textured surface")
    return np.concatenate(selected), np.concatenate(uv), np.concatenate(labels)


def ssim_map(reference, candidate):
    output = []
    for x, y in zip(reference, candidate):
        mx = gaussian_filter(x, 1.5, mode="nearest"); my = gaussian_filter(y, 1.5, mode="nearest")
        vx = gaussian_filter(x * x, 1.5, mode="nearest") - mx * mx
        vy = gaussian_filter(y * y, 1.5, mode="nearest") - my * my
        cov = gaussian_filter(x * y, 1.5, mode="nearest") - mx * my
        output.append(np.clip(((2 * mx * my + .01**2) * (2 * cov + .03**2)) /
                              np.maximum((mx * mx + my * my + .01**2) *
                                         (vx + vy + .03**2), 1e-12), -1, 1))
    return np.stack(output)


def patch_metrics(clean, degraded, face_patch, samples):
    clean_mesh, clean_rgb, _, clean_textured, clean_faces = clean
    degraded_mesh, degraded_rgb, _, degraded_textured, degraded_faces = degraded
    if len(clean_mesh.faces) != len(degraded_mesh.faces):
        raise ValueError("Texture attack changed face count")
    reference = clean_rgb.astype(np.float32) / 255.; candidate = degraded_rgb.astype(np.float32) / 255.
    difference = reference - candidate; common = clean_textured & degraded_textured
    luma = np.asarray([.2126, .7152, .0722], np.float32)
    ref_luma = reference @ luma; deg_luma = candidate @ luma
    gradient = lambda values: np.sqrt(
        np.stack([sobel(view, 1, mode="nearest") / 8 for view in values]) ** 2 +
        np.stack([sobel(view, 0, mode="nearest") / 8 for view in values]) ** 2)
    gradient_difference = np.abs(gradient(ref_luma) - gradient(deg_luma))
    structural = ssim_map(ref_luma, deg_luma)
    selected, uv, sample_labels = samples
    clean_surface, clean_valid = sample_texture_colors(clean_mesh, selected, uv)
    degraded_surface, degraded_valid = sample_texture_colors(degraded_mesh, selected, uv)
    surface_difference = clean_surface - degraded_surface
    surface_valid = clean_valid & degraded_valid
    count = int(face_patch.max()) + 1
    values = np.full((count, len(METRICS)), np.nan, np.float32)
    visible_pixels = np.zeros(count, np.int64); surface_counts = np.zeros(count, np.int64)
    maximum = np.zeros(count, np.float32)
    same_face = (clean_faces == degraded_faces) & (clean_faces >= 0)
    for patch in range(count):
        visible = common & same_face & (face_patch[np.maximum(clean_faces, 0)] == patch)
        visible_pixels[patch] = int(visible.sum())
        surface = (sample_labels == patch) & surface_valid
        surface_counts[patch] = int(surface.sum())
        if surface.any():
            delta = surface_difference[surface]
            values[patch, 4] = np.sqrt(np.mean(delta * delta))
            values[patch, 5] = np.mean(np.max(np.abs(delta), axis=1) > (1 / 255 + 1e-8))
            maximum[patch] = float(np.max(np.abs(delta)))
        if visible.any():
            rgb = difference[visible]
            values[patch, 0] = np.sqrt(np.mean(rgb * rgb))
            eroded = np.stack([binary_erosion(view, iterations=2) for view in visible])
            gradient_mask = eroded if eroded.any() else visible
            values[patch, 1] = np.mean(gradient_difference[gradient_mask])
            values[patch, 2] = 1.0 - np.mean(structural[visible])
            values[patch, 3] = np.mean(np.abs(reference[visible].mean(0) - candidate[visible].mean(0)))
            maximum[patch] = max(maximum[patch], float(np.max(np.abs(rgb))))
    return values, visible_pixels, surface_counts, maximum


def quality(metrics, specification):
    result = np.ones(len(metrics), np.float32)
    for patch, row in enumerate(metrics):
        burden = 0.; observed_weight = 0.
        for index, name in enumerate(METRICS):
            if np.isfinite(row[index]):
                setting = specification[name]; weight = float(setting["weight"])
                burden += weight * max(float(row[index]), 0.) / float(setting["scale"])
                observed_weight += weight
        result[patch] = np.exp(-burden / max(observed_weight, 1e-12))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--patch-map-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-size", type=int, default=224)
    parser.add_argument("--samples-per-patch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--asset-ids", nargs="*")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(args.dataset_manifest.read_text())["records"]
    requested = set(args.asset_ids or [])
    eligible = [row for row in manifest if row["attack"] in ATTACKS | {"clean"}
                and (not requested or row["asset_id"] in requested)]
    assets = sorted({row["asset_id"] for row in eligible})
    shard = set(assets[args.shard_index::args.shard_count])
    records = [row for row in eligible if row["asset_id"] in shard]
    clean_rows = {row["asset_id"]: row for row in records if row["attack"] == "clean"}
    specification = json.loads(args.config.read_text())["texture_fidelity"]
    rows = []
    for asset_index, (asset_id, clean_row) in enumerate(clean_rows.items()):
        clean = render(resolve_path(clean_row["gltf_path"], args.data_root), args.render_size)
        with np.load(args.patch_map_dir / f"{asset_id}.npz") as sidecar:
            face_patch = sidecar["face_patch"].astype(np.int64)
            patch_mask = sidecar["patch_mask"].astype(bool)
        samples = sample_patch_uv(clean[0], face_patch, args.samples_per_patch,
                                  stable_seed(args.seed, asset_id))
        asset_records = [row for row in records if row["asset_id"] == asset_id]
        for record in asset_records:
            if record["attack"] == "clean":
                metric = np.zeros((len(patch_mask), len(METRICS)), np.float32)
                visible = np.asarray([np.sum(clean[3] & (clean[4] >= 0) &
                                  (face_patch[np.maximum(clean[4], 0)] == patch))
                                      for patch in range(len(patch_mask))], np.int64)
                surface = np.bincount(samples[2], minlength=len(patch_mask))
                maximum = np.zeros(len(patch_mask), np.float32)
            else:
                attacked = render(resolve_path(record["gltf_path"], args.data_root), args.render_size)
                metric, visible, surface, maximum = patch_metrics(clean, attacked, face_patch, samples)
            raw = quality(metric, specification); noop = maximum <= 1e-12
            rows.append({"record": record, "metrics": metric, "raw": raw, "quality": raw.copy(),
                         "patch_mask": patch_mask, "visible": visible, "surface": surface,
                         "noop": noop})
        print(f"asset {asset_index + 1}/{len(clean_rows)} {asset_id}", flush=True)
    groups = {}
    for row in rows:
        if row["record"]["attack"] != "clean":
            groups.setdefault((row["record"]["asset_id"], row["record"]["attack"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: LEVEL_ORDER[row["record"]["level"]]); burden = np.zeros(16)
        for row in group:
            observable = ~row["noop"]
            burden[observable] = np.maximum(burden[observable], 1 - row["raw"][observable])
            row["quality"][observable] = 1 - burden[observable]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test", "blind"):
        selected = [row for row in rows if row["record"]["split"] == split]
        shape = (0, 16)
        np.savez_compressed(args.output_dir / f"patch_texture_targets_{split}.npz",
            asset_ids=np.asarray([r["record"]["asset_id"] for r in selected]),
            attacks=np.asarray([r["record"]["attack"] for r in selected]),
            levels=np.asarray([r["record"]["level"] for r in selected]),
            patch_texture_quality=np.stack([r["quality"] for r in selected]) if selected else np.empty(shape),
            patch_texture_quality_raw=np.stack([r["raw"] for r in selected]) if selected else np.empty(shape),
            patch_metrics=np.stack([r["metrics"] for r in selected]) if selected else np.empty((*shape, len(METRICS))),
            metric_names=np.asarray(METRICS),
            patch_mask=np.stack([r["patch_mask"] for r in selected]) if selected else np.empty(shape, bool),
            visible_pixel_count=np.stack([r["visible"] for r in selected]) if selected else np.empty(shape, np.int64),
            surface_sample_count=np.stack([r["surface"] for r in selected]) if selected else np.empty(shape, np.int64),
            objective_noop=np.stack([r["noop"] for r in selected]) if selected else np.empty(shape, bool))
    metadata = {"schema_version": 1, "seed": args.seed, "assets": len(clean_rows),
                "records": len(rows), "patches": 16, "metric_names": METRICS,
                "render_size": args.render_size, "samples_per_patch": args.samples_per_patch,
                "shard_index": args.shard_index, "shard_count": args.shard_count,
                "supervision": "full-reference offline Patch texture targets; inference remains no-reference",
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETE", **metadata}))


if __name__ == "__main__":
    main()
