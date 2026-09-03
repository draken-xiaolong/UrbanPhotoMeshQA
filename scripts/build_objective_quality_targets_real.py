#!/usr/bin/env python3
"""Build full-reference objective supervision for no-reference real-glTF QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


GEOMETRY_ATTACKS = {"geometry_hole", "mesh_simplification_qem", "geometry_noise_spike"}
TEXTURE_ATTACKS = {"texture_detail_loss", "texture_region_missing", "texture_misalignment"}
GEOMETRY_NAMES = ("chamfer_l2", "hausdorff", "normal_error", "missing_fraction",
                  "face_ratio_error", "topology_l1")
TEXTURE_NAMES = ("rgb_mae", "rgb_rmse", "gradient_mae", "color_mean_shift", "detail_energy_loss")


def key(row):
    return row["asset_id"], row["attack"], row["level"]


def geometry_metrics(clean, degraded):
    cp, dp = clean["points"][:, :3], degraded["points"][:, :3]
    cn, dn = clean["points"][:, 3:6], degraded["points"][:, 3:6]
    dtree, ctree = cKDTree(dp), cKDTree(cp)
    c2d, cindex = dtree.query(cp, k=1); d2c, _ = ctree.query(dp, k=1)
    normal_error = 1.0 - np.abs(np.sum(cn * dn[cindex], axis=1))
    chamfer = 0.5 * (np.mean(c2d ** 2) + np.mean(d2c ** 2))
    hausdorff = max(float(np.max(c2d)), float(np.max(d2c)))
    missing = float(np.mean(c2d > 0.02))
    face_ratio = abs(len(degraded["face_features"]) / max(len(clean["face_features"]), 1) - 1.0)
    topology = float(np.mean(np.abs(degraded["topology"] - clean["topology"])))
    return np.asarray([chamfer, hausdorff, np.mean(normal_error), missing, face_ratio, topology], np.float32)


def gradients(images):
    images = images.astype(np.float32) / 255.0
    gx = np.diff(images, axis=2, append=images[:, :, -1:, :])
    gy = np.diff(images, axis=1, append=images[:, -1:, :, :])
    return np.sqrt(gx * gx + gy * gy)


def texture_metrics(clean, degraded):
    a, b = clean["render_views"].astype(np.float32) / 255.0, degraded["render_views"].astype(np.float32) / 255.0
    difference = a - b
    ga, gb = gradients(clean["render_views"]), gradients(degraded["render_views"])
    return np.asarray([
        np.mean(np.abs(difference)), np.sqrt(np.mean(difference * difference)),
        np.mean(np.abs(ga - gb)), np.mean(np.abs(a.mean(axis=(1, 2)) - b.mean(axis=(1, 2)))),
        np.mean(np.abs(ga.mean(axis=(1, 2, 3)) - gb.mean(axis=(1, 2, 3)))),
    ], np.float32)


def load_cache(path):
    with np.load(path) as values:
        return {name: values[name].copy() for name in
                ("points", "face_features", "topology", "render_views", "patches", "patch_mask")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    targets = {key(row): row for row in dataset["records"]}
    cache_records = json.loads(args.cache_audit.read_text(encoding="utf-8"))["records"]
    cache_paths = {key(row): row["cache_path"] for row in cache_records}
    clean = {row["asset_id"]: load_cache(row["cache_path"])
             for row in cache_records if row["attack"] == "clean"}
    train_clean_patches = np.concatenate([
        value["patches"][value["patch_mask"]] for asset_id, value in clean.items()
        if targets[(asset_id, "clean", "clean")]["split"] == "train"
    ], axis=0)
    patch_feature_scale = np.maximum(train_clean_patches.std(axis=0), 1e-5)
    rows = []
    for index, record in enumerate(cache_records):
        target = targets[key(record)]; attack = record["attack"]
        geometry = np.zeros(len(GEOMETRY_NAMES), np.float32)
        texture = np.zeros(len(TEXTURE_NAMES), np.float32)
        local_burden_raw = np.zeros(16, np.float32)
        local_mask = clean[record["asset_id"]]["patch_mask"].copy()
        if attack in GEOMETRY_ATTACKS:
            degraded = load_cache(record["cache_path"])
            geometry = geometry_metrics(clean[record["asset_id"]], degraded)
            local_mask = degraded["patch_mask"].copy()
            clean_patch = clean[record["asset_id"]]["patches"][clean[record["asset_id"]]["patch_mask"]]
            degraded_patch = degraded["patches"][local_mask]
            nearest = cKDTree(clean_patch[:, :3]).query(degraded_patch[:, :3], k=1)[1]
            local_burden_raw[local_mask] = np.linalg.norm(
                (degraded_patch - clean_patch[nearest]) / patch_feature_scale, axis=1
            ) / np.sqrt(degraded_patch.shape[1])
        elif attack in TEXTURE_ATTACKS:
            texture = texture_metrics(clean[record["asset_id"]], load_cache(record["cache_path"]))
        rows.append({"target": target, "geometry": geometry, "texture": texture,
                     "local_burden_raw": local_burden_raw, "local_mask": local_mask})
        if index == 0 or (index + 1) % 100 == 0:
            print(f"metrics {index + 1}/{len(cache_records)}", flush=True)
    train = [row for row in rows if row["target"]["split"] == "train"]
    geometry_scale = np.maximum(np.quantile(np.stack([row["geometry"] for row in train]), 0.95, axis=0), 1e-8)
    texture_scale = np.maximum(np.quantile(np.stack([row["texture"] for row in train]), 0.95, axis=0), 1e-8)
    local_values = np.concatenate([row["local_burden_raw"][row["local_mask"]] for row in train
                                   if row["target"]["attack"] in GEOMETRY_ATTACKS])
    local_scale = max(float(np.quantile(local_values, 0.95)), 1e-8)
    for row in rows:
        geometry_burden = float(np.mean(np.clip(row["geometry"] / geometry_scale, 0.0, 1.0)))
        texture_burden = float(np.mean(np.clip(row["texture"] / texture_scale, 0.0, 1.0)))
        row["geometry_quality"] = 1.0 - geometry_burden
        row["texture_quality"] = 1.0 - texture_burden
        row["overall_quality"] = 1.0 - max(geometry_burden, texture_burden)
        row["patch_quality"] = 1.0 - np.clip(row["local_burden_raw"] / local_scale, 0.0, 1.0)
    monotonic_raw = {}
    for attack in sorted(GEOMETRY_ATTACKS | TEXTURE_ATTACKS):
        groups = {}
        for row in rows:
            if row["target"]["attack"] == attack:
                groups.setdefault(row["target"]["asset_id"], {})[row["target"]["level"]] = row["overall_quality"]
        checks = []
        for values in groups.values():
            checks.append(values["light"] >= values["medium"] - 1e-8
                          and values["medium"] >= values["heavy"] - 1e-8)
        monotonic_raw[attack] = {"passed": int(sum(checks)), "total": len(checks),
                                 "rate": float(np.mean(checks))}
        # Controlled light/medium/heavy attacks must not receive contradictory
        # targets because of finite surface sampling or resampling aliasing.
        # Preserve all raw objective vectors, but calibrate the scalar burden
        # with a cumulative monotonic envelope within each asset and attack.
        for asset_id in groups:
            triplet = [row for row in rows if row["target"]["asset_id"] == asset_id
                       and row["target"]["attack"] == attack]
            triplet.sort(key=lambda row: {"light": 0, "medium": 1, "heavy": 2}[row["target"]["level"]])
            burdens = np.maximum.accumulate([1.0 - row["overall_quality"] for row in triplet])
            for row, burden in zip(triplet, burdens):
                row["overall_quality"] = 1.0 - float(burden)
                if attack in GEOMETRY_ATTACKS:
                    row["geometry_quality"] = row["overall_quality"]
                else:
                    row["texture_quality"] = row["overall_quality"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test", "blind"):
        selected = [row for row in rows if row["target"]["split"] == split]
        np.savez_compressed(
            args.output_dir / f"objective_targets_{split}.npz",
            asset_ids=np.asarray([row["target"]["asset_id"] for row in selected]),
            attacks=np.asarray([row["target"]["attack"] for row in selected]),
            levels=np.asarray([row["target"]["level"] for row in selected]),
            geometry=np.stack([row["geometry"] for row in selected]),
            texture=np.stack([row["texture"] for row in selected]),
            geometry_quality=np.asarray([row["geometry_quality"] for row in selected], np.float32),
            texture_quality=np.asarray([row["texture_quality"] for row in selected], np.float32),
            overall_quality=np.asarray([row["overall_quality"] for row in selected], np.float32),
            patch_quality=np.stack([row["patch_quality"] for row in selected]).astype(np.float32),
            patch_mask=np.stack([row["local_mask"] for row in selected]),
        )
    metadata = {"schema_version": 1, "supervision": "full-reference objective metrics; inference remains no-reference",
                "geometry_metrics": GEOMETRY_NAMES, "texture_metrics": TEXTURE_NAMES,
                "geometry_scale_train_p95": geometry_scale.tolist(),
                "texture_scale_train_p95": texture_scale.tolist(),
                "patch_feature_scale": patch_feature_scale.tolist(),
                "patch_burden_train_p95": local_scale,
                "patch_supervision": "geometry attacks and clean models; nearest clean patch by normalized center",
                "quality_formula": "1 - mean(clipped metric / train P95); overall uses maximum modality burden",
                "severity_monotonicity_raw": monotonic_raw,
                "severity_monotonicity_after_calibration": 1.0,
                "monotonic_calibration": "cumulative maximum objective burden per asset and attack",
                "counts": {split: sum(row["target"]["split"] == split for row in rows)
                           for split in ("train", "val", "test", "blind")}}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
