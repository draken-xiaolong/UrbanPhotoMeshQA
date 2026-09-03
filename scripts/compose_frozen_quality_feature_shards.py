#!/usr/bin/env python3
"""Compose frozen feature shards in the exact dataset-manifest order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    args = parser.parse_args()
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))["records"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in args.splits:
        shards = []
        for root in args.shard_dirs:
            path = root / f"features_{split}.npz"
            if path.is_file():
                with np.load(path) as values:
                    shards.append({name: values[name].copy() for name in values.files})
        if not shards:
            raise ValueError(f"No shards for {split}")
        names = tuple(shards[0])
        if any(tuple(item) != names for item in shards):
            raise ValueError(f"Array mismatch for {split}")
        rows = {}
        for shard in shards:
            count = len(shard["asset_ids"])
            for index in range(count):
                item = (str(shard["asset_ids"][index]), str(shard["attacks"][index]),
                        str(shard["levels"][index]))
                if item in rows:
                    raise ValueError(f"Duplicate feature key: {item}")
                rows[item] = {name: shard[name][index] for name in names}
        expected = [(row["asset_id"], row["attack"], row["level"])
                    for row in dataset if row["split"] == split]
        missing = [item for item in expected if item not in rows]
        if missing or len(rows) != len(expected):
            raise ValueError(f"Feature mismatch {split}: got={len(rows)} expected={len(expected)} missing={missing[:3]}")
        payload = {name: np.stack([rows[item][name] for item in expected]) for name in names}
        np.savez_compressed(args.output_dir / f"features_{split}.npz", **payload)
        report[split] = len(expected)
    metadata = {"schema_version": 1, "counts": report,
                "composed_from_shards": len(args.shard_dirs),
                "dataset_manifest": str(args.dataset_manifest)}
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
