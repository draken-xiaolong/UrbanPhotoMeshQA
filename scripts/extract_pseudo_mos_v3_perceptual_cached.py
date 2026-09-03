#!/usr/bin/env python3
"""Batch perceptual full-reference teacher from deterministic raw-cache views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


LAYERS = (1, 3, 8, 12)


def key(row):
    return row["asset_id"], row["attack"], row["level"]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    weights = MobileNet_V3_Small_Weights.DEFAULT
    features = mobilenet_v3_small(weights=weights).features.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))["records"]
    cache_rows = json.loads(args.cache_audit.read_text(encoding="utf-8"))["records"]
    caches = {key(row): Path(row["cache_path"]) for row in cache_rows}
    clean = {row["asset_id"]: caches[(row["asset_id"], "clean", "clean")]
             for row in dataset if row["attack"] == "clean"}
    rows = []
    for start in range(0, len(dataset), args.batch_size):
        batch = dataset[start:start + args.batch_size]
        clean_views, attacked_views = [], []
        for row in batch:
            with np.load(clean[row["asset_id"]]) as values:
                clean_views.append(values["render_views"].copy())
            with np.load(caches[key(row)]) as values:
                attacked_views.append(values["render_views"].copy())
        clean_np = np.stack(clean_views).astype(np.float32) / 255.0
        attacked_np = np.stack(attacked_views).astype(np.float32) / 255.0
        b, v, h, w, _ = clean_np.shape
        pair = np.concatenate([clean_np, attacked_np], axis=0)
        x = torch.from_numpy(pair).to(device).permute(0, 1, 4, 2, 3).reshape(2*b*v, 3, h, w)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        maps = []
        for index, layer in enumerate(features):
            x = layer(x)
            if index in LAYERS:
                maps.append(F.normalize(x, dim=1).reshape(2*b, v, x.shape[1], x.shape[2], x.shape[3]))
        clean_mask = np.any(np.abs(clean_np * 255.0 - 245.0) > 3.0, axis=-1)
        attacked_mask = np.any(np.abs(attacked_np * 255.0 - 245.0) > 3.0, axis=-1)
        union_np = clean_mask | attacked_mask
        intersection = (clean_mask & attacked_mask).sum(axis=(1, 2, 3))
        union_count = union_np.sum(axis=(1, 2, 3))
        union = torch.from_numpy(union_np).to(device).float().reshape(b*v, 1, h, w)
        layer_mean, layer_p95, layer_worst = [], [], []
        for fmap in maps:
            left, right = fmap[:b], fmap[b:]
            distance = ((left - right) ** 2).mean(dim=2)
            weight = F.interpolate(union, size=distance.shape[-2:], mode="area").reshape(
                b, v, distance.shape[-2], distance.shape[-1])
            per_view = (distance * weight).sum((-1, -2)) / weight.sum((-1, -2)).clamp_min(1e-6)
            layer_mean.append(per_view.mean(1))
            layer_worst.append(per_view.max(1).values)
            p95 = []
            for sample in range(b):
                visible = distance[sample][weight[sample] > 0.05]
                p95.append(torch.quantile(visible, 0.95) if visible.numel() else torch.tensor(0.0, device=device))
            layer_p95.append(torch.stack(p95))
        pixel_difference = (clean_np - attacked_np) ** 2
        for index, row in enumerate(batch):
            visible = union_np[index]
            rgb_rmse = float(np.sqrt(pixel_difference[index][visible].mean())) if visible.any() else 0.0
            rows.append({
                "record": row,
                "feature_mean": float(torch.stack(layer_mean)[:, index].mean()),
                "feature_p95_mean": float(torch.stack(layer_p95)[:, index].mean()),
                "feature_worst_view_mean": float(torch.stack(layer_worst)[:, index].mean()),
                "rgb_rmse_union": rgb_rmse,
                "silhouette_iou": float(intersection[index] / max(union_count[index], 1)),
            })
        print(f"teacher {min(start + len(batch), len(dataset))}/{len(dataset)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = ("feature_mean", "feature_p95_mean", "feature_worst_view_mean",
                    "rgb_rmse_union", "silhouette_iou")
    for split in ("train", "val", "test", "blind"):
        selected = [row for row in rows if row["record"]["split"] == split]
        np.savez_compressed(
            args.output_dir / f"perceptual_teacher_{split}.npz",
            asset_ids=np.asarray([row["record"]["asset_id"] for row in selected]),
            attacks=np.asarray([row["record"]["attack"] for row in selected]),
            levels=np.asarray([row["record"]["level"] for row in selected]),
            metrics=np.asarray([[row[name] for name in metric_names] for row in selected], np.float32),
            metric_names=np.asarray(metric_names),
        )
    metadata = {"schema_version": 1, "teacher": "MobileNetV3-Small cached aligned six-view",
                "layers": list(LAYERS), "input_cache_size": 96, "network_size": 224,
                "records": len(rows), "batch_size": args.batch_size}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", **metadata}))


if __name__ == "__main__":
    main()
