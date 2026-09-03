#!/usr/bin/env python3
"""Build aligned full-reference objective geometry-quality targets for cached queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from urbanphotomeshqa.gltf import GltfReader, sample_surface  # noqa: E402
from urbanphotomeshqa.mesh_attacks import apply_mesh_attack  # noqa: E402
from urbanphotomeshqa.quality import objective_quality_metrics  # noqa: E402


GEOMETRY_ATTACKS = {"connected_crop", "hole", "retriangulate"}
METRICS = (
    "chamfer_l2", "hausdorff", "normal_error", "missing_fraction",
    "outlier_fraction", "bbox_extent_relative_error",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def in_clean_frame(samples, stats, clean_stats):
    raw = samples[:, :3] * float(stats["normalization_scale"]) + np.asarray(stats["normalization_center"])
    xyz = (raw - np.asarray(clean_stats["normalization_center"])) / float(clean_stats["normalization_scale"])
    return np.concatenate([xyz, samples[:, 3:6]], axis=1).astype(np.float32)


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for split in ("train", "val", "test", "blind"):
        records = {row["asset_id"]: row for row in manifest["records"] if row["split"] == split}
        with np.load(args.feature_dir / f"scores_{split}.npz") as values:
            query_ids = values["query_ids"].astype(str)
            attacks = values["attacks"].astype(str)
            severities = values["severities"].astype(np.float32)
        selected_indices, targets, target_attacks, target_ids = [], [], [], []
        gallery_order = []
        for asset_id in query_ids.tolist():
            if asset_id not in gallery_order: gallery_order.append(asset_id)
        for asset_index, asset_id in enumerate(gallery_order):
            clean = GltfReader(PROJECT / records[asset_id]["gltf_path"]).load_mesh()
            clean_samples, clean_stats = sample_surface(clean, args.points, args.seed + asset_index * 1009)
            reference = in_clean_frame(clean_samples, clean_stats, clean_stats)
            indices = np.flatnonzero(query_ids == asset_id)
            geometry_local_index = 0
            for query_index in indices.tolist():
                attack = attacks[query_index]
                if attack not in GEOMETRY_ATTACKS:
                    continue
                geometry_local_index += 1
                attacked = apply_mesh_attack(
                    clean, attack, float(severities[query_index]),
                    args.seed + asset_index * 1009 + query_index,
                )
                degraded_samples, degraded_stats = sample_surface(
                    attacked, args.points, args.seed + asset_index * 1009 + geometry_local_index * 37
                )
                degraded = in_clean_frame(degraded_samples, degraded_stats, clean_stats)
                values = objective_quality_metrics(reference, degraded)
                selected_indices.append(query_index)
                targets.append([values[name] for name in METRICS])
                target_attacks.append(attack); target_ids.append(asset_id)
            if asset_index == 0 or (asset_index + 1) % 10 == 0:
                print(f"{split}: {asset_index + 1}/{len(gallery_order)}", flush=True)
        np.savez_compressed(
            args.output_dir / f"targets_{split}.npz",
            query_indices=np.asarray(selected_indices, np.int64), targets=np.asarray(targets, np.float32),
            attacks=np.asarray(target_attacks), asset_ids=np.asarray(target_ids), metrics=np.asarray(METRICS),
        )
    metadata = {
        "status": "OBJECTIVE_GEOMETRY_QUALITY_TARGETS_COMPLETE", "seed": args.seed,
        "points": args.points, "metrics": METRICS,
        "attacks": sorted(GEOMETRY_ATTACKS),
        "coordinate_frame": "both reference and degraded samples normalized by clean-object bbox frame",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
