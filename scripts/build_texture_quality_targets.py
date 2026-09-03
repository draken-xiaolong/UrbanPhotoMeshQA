#!/usr/bin/env python3
"""Build aligned full-reference objective texture-quality targets from rendered views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from urbanphotomeshqa.gltf import GltfReader  # noqa: E402
from urbanphotomeshqa.render_features import legacy_image_variant, render_views  # noqa: E402


TEXTURE_ATTACKS = {"jpeg50", "blur1.5", "brightness0.55", "downsample32", "occlusion20"}
METRICS = ("rgb_l1", "rgb_mse", "one_minus_ssim", "edge_l1", "color_mean_shift")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def global_ssim(reference, degraded):
    values = []
    for channel in range(3):
        x = reference[..., channel].reshape(-1); y = degraded[..., channel].reshape(-1)
        mx, my = x.mean(), y.mean(); vx, vy = x.var(), y.var()
        covariance = np.mean((x - mx) * (y - my))
        values.append(((2 * mx * my + 0.01**2) * (2 * covariance + 0.03**2)) /
                      ((mx**2 + my**2 + 0.01**2) * (vx + vy + 0.03**2)))
    return float(np.mean(values))


def texture_metrics(reference_views, degraded_views):
    reference = np.stack(reference_views).astype(np.float32) / 255.0
    degraded = np.stack(degraded_views).astype(np.float32) / 255.0
    difference = np.abs(reference - degraded)
    ref_edges = np.concatenate([np.diff(reference, axis=2).reshape(-1), np.diff(reference, axis=1).reshape(-1)])
    deg_edges = np.concatenate([np.diff(degraded, axis=2).reshape(-1), np.diff(degraded, axis=1).reshape(-1)])
    return {
        "rgb_l1": float(difference.mean()),
        "rgb_mse": float(np.mean((reference - degraded) ** 2)),
        "one_minus_ssim": float(1.0 - global_ssim(reference, degraded)),
        "edge_l1": float(np.mean(np.abs(ref_edges - deg_edges))),
        "color_mean_shift": float(np.mean(np.abs(reference.mean(axis=(0, 1, 2)) - degraded.mean(axis=(0, 1, 2))))),
    }


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for split in ("train", "val", "test", "blind"):
        records = {row["asset_id"]: row for row in manifest["records"] if row["split"] == split}
        with np.load(args.feature_dir / f"scores_{split}.npz") as values:
            query_ids = values["query_ids"].astype(str); attacks = values["attacks"].astype(str)
        gallery_order = []
        for asset_id in query_ids.tolist():
            if asset_id not in gallery_order: gallery_order.append(asset_id)
        selected_indices, targets, target_attacks, target_ids = [], [], [], []
        for asset_index, asset_id in enumerate(gallery_order):
            mesh = GltfReader(PROJECT / records[asset_id]["gltf_path"]).load_mesh(include_texture=True)
            clean_views = render_views(mesh, args.render_size)
            for query_index in np.flatnonzero(query_ids == asset_id).tolist():
                attack = attacks[query_index]
                if attack not in TEXTURE_ATTACKS: continue
                degraded_views = legacy_image_variant(clean_views, attack, args.seed + asset_index)
                metric = texture_metrics(clean_views, degraded_views)
                selected_indices.append(query_index); targets.append([metric[name] for name in METRICS])
                target_attacks.append(attack); target_ids.append(asset_id)
            if asset_index == 0 or (asset_index + 1) % 10 == 0:
                print(f"{split}: {asset_index + 1}/{len(gallery_order)}", flush=True)
        np.savez_compressed(args.output_dir / f"targets_{split}.npz",
                            query_indices=np.asarray(selected_indices, np.int64),
                            targets=np.asarray(targets, np.float32), attacks=np.asarray(target_attacks),
                            asset_ids=np.asarray(target_ids), metrics=np.asarray(METRICS))
    metadata = {"status": "OBJECTIVE_TEXTURE_QUALITY_TARGETS_COMPLETE", "seed": args.seed,
                "render_size": args.render_size, "metrics": METRICS, "attacks": sorted(TEXTURE_ATTACKS),
                "scope": "six aligned standard rendered views; background and camera changes excluded"}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
