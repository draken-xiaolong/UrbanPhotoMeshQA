#!/usr/bin/env python3
"""Build masked, material-aware full-reference texture targets from real glTF pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, sobel

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.texture import (
    STANDARD_DIRECTIONS,
    _load_textures,
    _material_render_profile,
    _wrap_texture_coordinate,
    render_textured_view_with_masks,
)


TEXTURE_ATTACKS = {
    "texture_detail_loss",
    "texture_region_missing",
    "texture_misalignment",
}
METRIC_NAMES = (
    "masked_rgb_mae",
    "masked_rgb_rmse",
    "masked_luma_mae",
    "masked_gradient_mae",
    "masked_one_minus_ssim",
    "masked_color_mean_shift",
    "masked_detail_energy_relative_change",
    "visible_changed_fraction_1_255",
    "surface_rgb_mae",
    "surface_rgb_rmse",
    "surface_changed_fraction_1_255",
)


def resolve_path(value: str, data_root: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if data_root is None:
        raise ValueError(f"Relative glTF path requires --data-root: {value}")
    return data_root / path


def render_asset(path: Path, size: int):
    mesh = GltfReader(path).load_mesh(include_texture=True)
    rendered = [
        render_textured_view_with_masks(mesh, direction=direction, size=size)
        for direction in STANDARD_DIRECTIONS
    ]
    return (
        mesh,
        np.stack([item[0] for item in rendered]),
        np.stack([item[1] for item in rendered]),
        np.stack([item[2] for item in rendered]),
    )


def stable_seed(seed: int, asset_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}|{asset_id}|surface-uv".encode()).digest()[:8], "little"
    )


def sample_surface_uv(mesh, count: int, seed: int):
    triangles = np.asarray(mesh.vertices, np.float64)[mesh.faces]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    textured_face = np.asarray([
        path is not None for path in mesh.metadata.get("material_texture_paths", [])
    ])
    materials = np.asarray(mesh.face_materials, np.int64)
    valid_material = (materials >= 0) & (materials < len(textured_face))
    valid = np.zeros(len(materials), dtype=bool)
    valid[valid_material] = textured_face[materials[valid_material]]
    valid &= np.isfinite(np.asarray(mesh.texcoords)[mesh.faces]).all(axis=(1, 2))
    weights = np.where(valid, double_area, 0.0)
    if weights.sum() <= 0:
        raise ValueError("Mesh has no non-degenerate textured surface")
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(mesh.faces), count, replace=True, p=weights)
    r1 = np.sqrt(rng.random((count, 1)))
    r2 = rng.random((count, 1))
    barycentric = np.concatenate([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1)
    face_uv = np.asarray(mesh.texcoords, np.float64)[mesh.faces[selected]]
    uv = np.sum(face_uv * barycentric[:, :, None], axis=1)
    return selected, uv


def sample_texture_colors(mesh, selected_faces, uv):
    textures = _load_textures(mesh)
    materials = np.asarray(mesh.face_materials, np.int64)[selected_faces]
    output = np.zeros((len(uv), 3), dtype=np.float32)
    valid = np.zeros(len(uv), dtype=bool)
    for material in np.unique(materials):
        indices = np.flatnonzero(materials == material)
        texture = textures[material] if 0 <= material < len(textures) else None
        if texture is None:
            continue
        profile = _material_render_profile(mesh, int(material))
        u = _wrap_texture_coordinate(uv[indices, 0], profile["wrap_s"])
        v = _wrap_texture_coordinate(uv[indices, 1], profile["wrap_t"])
        x = u * (texture.shape[1] - 1)
        y = (1.0 - v) * (texture.shape[0] - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, texture.shape[1] - 1)
        y1 = np.minimum(y0 + 1, texture.shape[0] - 1)
        wx = (x - x0)[:, None]
        wy = (y - y0)[:, None]
        top = (
            texture[y0, x0].astype(np.float32) / 255.0 * (1.0 - wx)
            + texture[y0, x1].astype(np.float32) / 255.0 * wx
        )
        bottom = (
            texture[y1, x0].astype(np.float32) / 255.0 * (1.0 - wx)
            + texture[y1, x1].astype(np.float32) / 255.0 * wx
        )
        sampled = top * (1.0 - wy) + bottom * wy
        factor = profile["factor"].astype(np.float32)
        output[indices] = np.clip(sampled[:, :3] * factor[:3], 0.0, 1.0)
        valid[indices] = sampled[:, 3] * factor[3] > 0.0
    return output, valid


def _masked_ssim(reference: np.ndarray, degraded: np.ndarray, mask: np.ndarray) -> float:
    # Wang et al. SSIM constants for normalized luminance, evaluated only on
    # valid textured pixels. Gaussian filtering never crosses between views.
    values = []
    weights = []
    for view in range(len(reference)):
        valid = mask[view]
        if not valid.any():
            continue
        x = reference[view]
        y = degraded[view]
        mx = gaussian_filter(x, 1.5, mode="nearest")
        my = gaussian_filter(y, 1.5, mode="nearest")
        vx = gaussian_filter(x * x, 1.5, mode="nearest") - mx * mx
        vy = gaussian_filter(y * y, 1.5, mode="nearest") - my * my
        covariance = gaussian_filter(x * y, 1.5, mode="nearest") - mx * my
        numerator = (2.0 * mx * my + 0.01**2) * (2.0 * covariance + 0.03**2)
        denominator = (mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2)
        score = np.clip(numerator / np.maximum(denominator, 1e-12), -1.0, 1.0)
        values.append(float(score[valid].mean()))
        weights.append(int(valid.sum()))
    if not weights:
        raise ValueError("No visible textured pixels in any standard view")
    return float(np.average(values, weights=weights))


def texture_metrics(clean, degraded, surface_samples):
    clean_mesh, clean_rgb, clean_foreground, clean_textured = clean
    degraded_mesh, degraded_rgb, degraded_foreground, degraded_textured = degraded
    mask = clean_textured & degraded_textured
    if not mask.any():
        raise ValueError("Clean/degraded pair has no common visible textured pixels")
    edge_mask = np.stack([binary_erosion(view, iterations=2) for view in mask])
    if not edge_mask.any():
        edge_mask = mask

    reference = clean_rgb.astype(np.float32) / 255.0
    candidate = degraded_rgb.astype(np.float32) / 255.0
    difference = reference - candidate
    luma_weights = np.asarray([0.2126, 0.7152, 0.0722], np.float32)
    ref_luma = reference @ luma_weights
    deg_luma = candidate @ luma_weights
    ref_gx = np.stack([sobel(view, axis=1, mode="nearest") / 8.0 for view in ref_luma])
    ref_gy = np.stack([sobel(view, axis=0, mode="nearest") / 8.0 for view in ref_luma])
    deg_gx = np.stack([sobel(view, axis=1, mode="nearest") / 8.0 for view in deg_luma])
    deg_gy = np.stack([sobel(view, axis=0, mode="nearest") / 8.0 for view in deg_luma])
    ref_gradient = np.sqrt(ref_gx * ref_gx + ref_gy * ref_gy)
    deg_gradient = np.sqrt(deg_gx * deg_gx + deg_gy * deg_gy)

    rgb_pixels = difference[mask]
    ref_color_mean = reference[mask].mean(axis=0)
    deg_color_mean = candidate[mask].mean(axis=0)
    ref_energy = float(ref_gradient[edge_mask].mean())
    deg_energy = float(deg_gradient[edge_mask].mean())
    changed = np.max(np.abs(difference), axis=-1) > (1.0 / 255.0 + 1e-8)
    selected_faces, uv = surface_samples
    clean_surface, clean_surface_valid = sample_texture_colors(clean_mesh, selected_faces, uv)
    degraded_surface, degraded_surface_valid = sample_texture_colors(degraded_mesh, selected_faces, uv)
    surface_valid = clean_surface_valid & degraded_surface_valid
    if not surface_valid.any():
        raise ValueError("Clean/degraded pair has no common sampled textured surface")
    surface_difference = clean_surface[surface_valid] - degraded_surface[surface_valid]
    surface_changed = np.max(np.abs(surface_difference), axis=1) > (1.0 / 255.0 + 1e-8)
    metrics = np.asarray([
        np.mean(np.abs(rgb_pixels)),
        np.sqrt(np.mean(rgb_pixels * rgb_pixels)),
        np.mean(np.abs((ref_luma - deg_luma)[mask])),
        np.mean(np.abs((ref_gradient - deg_gradient)[edge_mask])),
        1.0 - _masked_ssim(ref_luma, deg_luma, mask),
        np.mean(np.abs(ref_color_mean - deg_color_mean)),
        abs(ref_energy - deg_energy) / max(ref_energy, 1e-6),
        np.mean(changed[mask]),
        np.mean(np.abs(surface_difference)),
        np.sqrt(np.mean(surface_difference * surface_difference)),
        np.mean(surface_changed),
    ], dtype=np.float32)
    coverage = {
        "common_textured_pixels": int(mask.sum()),
        "clean_foreground_pixels": int(clean_foreground.sum()),
        "clean_textured_pixels": int(clean_textured.sum()),
        "degraded_foreground_pixels": int(degraded_foreground.sum()),
        "degraded_textured_pixels": int(degraded_textured.sum()),
        "clean_textured_foreground_fraction": float(
            clean_textured.sum() / max(int(clean_foreground.sum()), 1)
        ),
        "maximum_rgb_difference": int(
            np.max(np.abs(clean_rgb.astype(np.int16) - degraded_rgb.astype(np.int16)))
        ),
        "maximum_surface_rgb_difference": float(np.max(np.abs(surface_difference))),
    }
    return metrics, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-size", type=int, default=224)
    parser.add_argument("--surface-samples", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--asset-ids", nargs="*")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")

    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    requested = set(args.asset_ids or [])
    eligible = [
        row for row in manifest["records"]
        if row["attack"] in TEXTURE_ATTACKS | {"clean"}
        and (not requested or row["asset_id"] in requested)
    ]
    found = {row["asset_id"] for row in eligible}
    if requested and requested != found:
        raise ValueError(f"Unknown requested assets: {sorted(requested - found)}")
    ordered_assets = sorted(found)
    shard_assets = set(ordered_assets[args.shard_index::args.shard_count])
    records = [row for row in eligible if row["asset_id"] in shard_assets]
    clean_records = {
        row["asset_id"]: row for row in records if row["attack"] == "clean"
    }
    clean_renders = {
        asset_id: render_asset(resolve_path(row["gltf_path"], args.data_root), args.render_size)
        for asset_id, row in clean_records.items()
    }
    surface_samples = {
        asset_id: sample_surface_uv(
            clean[0], args.surface_samples, stable_seed(args.seed, asset_id)
        )
        for asset_id, clean in clean_renders.items()
    }

    rows = []
    for index, record in enumerate(records):
        clean = clean_renders[record["asset_id"]]
        if record["attack"] == "clean":
            metrics = np.zeros(len(METRIC_NAMES), np.float32)
            foreground = int(clean[2].sum())
            textured = int(clean[3].sum())
            coverage = {
                "common_textured_pixels": textured,
                "clean_foreground_pixels": foreground,
                "clean_textured_pixels": textured,
                "degraded_foreground_pixels": foreground,
                "degraded_textured_pixels": textured,
                "clean_textured_foreground_fraction": textured / max(foreground, 1),
                "maximum_rgb_difference": 0,
                "maximum_surface_rgb_difference": 0.0,
            }
        else:
            degraded = render_asset(
                resolve_path(record["gltf_path"], args.data_root), args.render_size
            )
            metrics, coverage = texture_metrics(
                clean, degraded, surface_samples[record["asset_id"]]
            )
        rows.append((record, metrics, coverage))
        print(
            f"target {index + 1}/{len(records)} {record['asset_id']} "
            f"{record['attack']} {record['level']}", flush=True
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test", "blind"):
        selected = [row for row in rows if row[0]["split"] == split]
        metrics = np.stack([row[1] for row in selected]) if selected else np.empty(
            (0, len(METRIC_NAMES)), dtype=np.float32
        )
        np.savez_compressed(
            args.output_dir / f"texture_targets_v2_{split}.npz",
            asset_ids=np.asarray([row[0]["asset_id"] for row in selected]),
            attacks=np.asarray([row[0]["attack"] for row in selected]),
            levels=np.asarray([row[0]["level"] for row in selected]),
            metrics=metrics,
            metric_names=np.asarray(METRIC_NAMES),
            common_textured_pixels=np.asarray(
                [row[2]["common_textured_pixels"] for row in selected], np.int64
            ),
            textured_foreground_fraction=np.asarray(
                [row[2]["clean_textured_foreground_fraction"] for row in selected], np.float32
            ),
            maximum_rgb_difference=np.asarray(
                [row[2]["maximum_rgb_difference"] for row in selected], np.int16
            ),
            objective_noop=np.asarray(
                [row[2]["maximum_rgb_difference"] == 0
                 and row[2]["maximum_surface_rgb_difference"] == 0.0
                 for row in selected], bool
            ),
        )

    source_files = [Path(__file__), Path(__file__).parents[1] / "src/urbanphotomeshqa/texture.py"]
    implementation_sha256 = hashlib.sha256(
        b"".join(path.read_bytes() for path in source_files)
    ).hexdigest()
    metadata = {
        "schema_version": 2,
        "method": (
            "six aligned material-aware orthographic views with explicit textured masks, "
            "plus deterministic area-weighted triangle-to-UV surface samples"
        ),
        "surface_uv_samples_per_asset": args.surface_samples,
        "seed": args.seed,
        "render_size": args.render_size,
        "directions": STANDARD_DIRECTIONS,
        "metric_names": METRIC_NAMES,
        "scope": "pilot" if requested else "formal",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assets": len(clean_records),
        "records": len(rows),
        "dataset_manifest": str(args.dataset_manifest),
        "implementation_sha256": implementation_sha256,
        "dependencies": {
            "SSIM": "Wang et al., IEEE TIP 2004; local implementation",
            "SciPy": "BSD-3-Clause",
            "NumPy": "BSD-3-Clause",
            "Pillow": "HPND",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETE", **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
