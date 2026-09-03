#!/usr/bin/env python3
"""Compose texture-target shards in the exact formal manifest order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TEXTURE_ATTACKS = {
    "texture_detail_loss",
    "texture_region_missing",
    "texture_misalignment",
}
ARRAYS = (
    "metrics",
    "common_textured_pixels",
    "textured_foreground_fraction",
    "maximum_rgb_difference",
    "objective_noop",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))["records"]
    values_by_key = {}
    metric_names = None
    metadata = []
    for root in args.shard_dirs:
        metadata.append(json.loads((root / "metadata.json").read_text(encoding="utf-8")))
        for split in ("train", "val", "test", "blind"):
            with np.load(root / f"texture_targets_v2_{split}.npz") as values:
                current_names = values["metric_names"].astype(str).tolist()
                if metric_names is None:
                    metric_names = current_names
                elif metric_names != current_names:
                    raise ValueError(f"Metric mismatch in {root}/{split}")
                keys = list(zip(
                    values["asset_ids"].astype(str),
                    values["attacks"].astype(str),
                    values["levels"].astype(str),
                ))
                for index, key in enumerate(keys):
                    if key in values_by_key:
                        raise ValueError(f"Duplicate target key: {key}")
                    values_by_key[key] = {name: values[name][index].copy() for name in ARRAYS}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_total = 0
    for split in ("train", "val", "test", "blind"):
        rows = [
            row for row in manifest
            if row["split"] == split and row["attack"] in TEXTURE_ATTACKS | {"clean"}
        ]
        keys = [(row["asset_id"], row["attack"], row["level"]) for row in rows]
        missing = [key for key in keys if key not in values_by_key]
        if missing:
            raise ValueError(f"Missing {len(missing)} keys in {split}; first={missing[0]}")
        expected_total += len(keys)
        payload = {
            "asset_ids": np.asarray([key[0] for key in keys]),
            "attacks": np.asarray([key[1] for key in keys]),
            "levels": np.asarray([key[2] for key in keys]),
            "metric_names": np.asarray(metric_names),
        }
        for name in ARRAYS:
            payload[name] = np.stack([values_by_key[key][name] for key in keys])
        np.savez_compressed(args.output_dir / f"texture_targets_v2_{split}.npz", **payload)

    if len(values_by_key) != expected_total:
        raise ValueError(f"Unexpected shard keys: got {len(values_by_key)}, expected {expected_total}")
    report = {
        **metadata[0],
        "scope": "formal",
        "records": expected_total,
        "assets": len({key[0] for key in values_by_key}),
        "composed_from_shards": len(args.shard_dirs),
        "shard_metadata": metadata,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETE", "records": expected_total,
                      "assets": report["assets"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
