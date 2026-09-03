#!/usr/bin/env python3
"""Compose audited spatial texture feature shards in formal manifest order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TEXTURE_ATTACKS = {"clean", "texture_detail_loss", "texture_region_missing",
                   "texture_misalignment"}


def key(row) -> tuple[str, str, str]:
    return str(row["asset_id"]), str(row["attack"]), str(row["level"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, nargs="+", required=True)
    parser.add_argument("--objective-target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in args.audit]
    if not all(audit["passed"] for audit in audits):
        raise ValueError("At least one feature shard failed")
    signatures = {audit["signature"] for audit in audits}
    if len(signatures) != 1:
        raise ValueError(f"Feature signature mismatch: {signatures}")
    records = [row for audit in audits for row in audit["records"]]
    lookup = {key(row): row for row in records}
    if len(lookup) != len(records):
        raise ValueError("Duplicate feature record keys")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = [row for row in manifest["records"]
                if row["split"] in {"train", "val"} and row["attack"] in TEXTURE_ATTACKS]
    missing = [key(row) for row in expected if key(row) not in lookup]
    extra = [item for item in lookup if item not in {key(row) for row in expected}]
    if missing or extra:
        raise ValueError(f"Feature coverage mismatch: missing={missing[:3]} extra={extra[:3]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shapes = None
    counts = {}
    for split in ("train", "val"):
        rows = [row for row in expected if row["split"] == split]
        targets_path = args.objective_target_dir / f"objective_targets_v2_{split}.npz"
        with np.load(targets_path) as targets:
            target_lookup = {item: index for index, item in enumerate(zip(
                targets["asset_ids"].astype(str), targets["attacks"].astype(str),
                targets["levels"].astype(str)))}
            target_indices = np.asarray([target_lookup[key(row)] for row in rows], np.int64)
            texture_quality = targets["texture_quality"][target_indices].astype(np.float32)
        arrays = {name: [] for name in ("tokens", "token_mask", "view_stats", "asset_stats")}
        for row in rows:
            with np.load(lookup[key(row)]["feature_path"]) as values:
                for name in arrays:
                    arrays[name].append(values[name].copy())
        arrays = {name: np.stack(values) for name, values in arrays.items()}
        if shapes is None:
            shapes = {name: list(value.shape[1:]) for name, value in arrays.items()}
        np.savez_compressed(
            args.output_dir / f"features_{split}.npz",
            asset_ids=np.asarray([row["asset_id"] for row in rows]),
            attacks=np.asarray([row["attack"] for row in rows]),
            levels=np.asarray([row["level"] for row in rows]),
            texture_quality=texture_quality,
            **arrays,
        )
        counts[split] = len(rows)
    metadata = {"schema_version": 1, "status": "COMPLETE", "signature": signatures.pop(),
                "counts": counts, "shapes": shapes, "loaded_splits": ["train", "val"],
                "test_blind_loaded": False, "attacks": sorted(TEXTURE_ATTACKS)}
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
