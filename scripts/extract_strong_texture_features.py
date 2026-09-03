#!/usr/bin/env python3
"""Extract stronger EfficientNet-B0 six-view texture features in GPU batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def key(row):
    return row["asset_id"], row["attack"], row["level"]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    weights = EfficientNet_B0_Weights.DEFAULT
    network = efficientnet_b0(weights=weights).to(device).eval()
    transform = weights.transforms()
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))["records"]
    targets = {key(row): row for row in dataset if row["split"] in set(args.splits)}
    audit = json.loads(args.cache_audit.read_text(encoding="utf-8"))["records"]
    selected = [row for row in audit if key(row) in targets]
    output = {split: [] for split in args.splits}
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        views = []
        for row in batch:
            with np.load(row["cache_path"]) as raw:
                views.extend(raw["render_views"].copy())
        tensor = torch.stack([transform(Image.fromarray(view)) for view in views]).to(device)
        embedding = F.normalize(network.avgpool(network.features(tensor)).flatten(1), dim=1)
        embedding = embedding.reshape(len(batch), 6, -1)
        pooled = F.normalize(embedding.mean(1), dim=1).cpu().numpy().astype(np.float32)
        for index, row in enumerate(batch):
            target = targets[key(row)]
            output[target["split"]].append((
                target, pooled[index], embedding[index].cpu().numpy().astype(np.float32)
            ))
        print(f"texture {min(start + len(batch), len(selected))}/{len(selected)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        np.savez_compressed(
            args.output_dir / f"strong_texture_{split}.npz",
            asset_ids=np.asarray([row[0]["asset_id"] for row in rows]),
            attacks=np.asarray([row[0]["attack"] for row in rows]),
            levels=np.asarray([row[0]["level"] for row in rows]),
            texture=np.stack([row[1] for row in rows]),
            texture_views=np.stack([row[2] for row in rows]),
        )
    metadata = {"schema_version": 1, "encoder": "ImageNet EfficientNet-B0 frozen",
                "dimension": int(next(iter(output.values()))[0][1].shape[0]),
                "counts": {split: len(rows) for split, rows in output.items()}}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", **metadata}))


if __name__ == "__main__":
    main()
