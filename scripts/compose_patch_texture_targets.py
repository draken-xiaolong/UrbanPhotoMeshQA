#!/usr/bin/env python3
"""Compose and audit Patch texture-target shards in formal manifest order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ATTACKS = {"texture_detail_loss", "texture_region_missing", "texture_misalignment"}
ARRAYS = ("patch_texture_quality", "patch_texture_quality_raw", "patch_metrics", "patch_mask",
          "visible_pixel_count", "surface_sample_count", "objective_noop")
LEVEL = {"light": 0, "medium": 1, "heavy": 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.dataset_manifest.read_text())["records"]
    lookup = {}; metric_names = None; shard_metadata = []
    for root in args.shard_dirs:
        shard_metadata.append(json.loads((root / "metadata.json").read_text()))
        for split in ("train", "val", "test", "blind"):
            with np.load(root / f"patch_texture_targets_{split}.npz") as values:
                names = values["metric_names"].astype(str).tolist()
                if metric_names is None: metric_names = names
                if metric_names != names: raise ValueError(f"Metric mismatch: {root}/{split}")
                keys = zip(values["asset_ids"].astype(str), values["attacks"].astype(str),
                           values["levels"].astype(str))
                for index, key in enumerate(keys):
                    if key in lookup: raise ValueError(f"Duplicate key: {key}")
                    lookup[key] = {name: values[name][index].copy() for name in ARRAYS}
    args.output_dir.mkdir(parents=True, exist_ok=True); ordered = []
    counts = {}
    for split in ("train", "val", "test", "blind"):
        records = [row for row in manifest if row["split"] == split
                   and row["attack"] in ATTACKS | {"clean"}]
        keys = [(row["asset_id"], row["attack"], row["level"]) for row in records]
        missing = [key for key in keys if key not in lookup]
        if missing: raise ValueError(f"Missing {len(missing)} {split} keys; first={missing[0]}")
        counts[split] = len(keys); ordered.extend(keys)
        payload = {"asset_ids": np.asarray([key[0] for key in keys]),
                   "attacks": np.asarray([key[1] for key in keys]),
                   "levels": np.asarray([key[2] for key in keys]),
                   "metric_names": np.asarray(metric_names)}
        for name in ARRAYS: payload[name] = np.stack([lookup[key][name] for key in keys])
        payload["texture_supervision_mask"] = (
            (payload["visible_pixel_count"] > 0) | (payload["surface_sample_count"] > 0)
        ) & payload["patch_mask"]
        np.savez_compressed(args.output_dir / f"patch_texture_targets_{split}.npz", **payload)
    if set(ordered) != set(lookup): raise ValueError("Shard output contains non-formal keys")
    clean_ok = all(np.allclose(lookup[key]["patch_texture_quality"], 1)
                   for key in ordered if key[1] == "clean")
    finite = all(np.isfinite(lookup[key]["patch_texture_quality"]).all() for key in ordered)
    supervised = np.stack([
        ((lookup[key]["visible_pixel_count"] > 0) |
         (lookup[key]["surface_sample_count"] > 0)) & lookup[key]["patch_mask"]
        for key in ordered])
    monotonic_passed = monotonic_total = 0
    for asset_id in sorted({key[0] for key in ordered}):
        for attack in sorted(ATTACKS):
            group = sorted([key for key in ordered if key[0] == asset_id and key[1] == attack],
                           key=lambda key: LEVEL[key[2]])
            for patch in range(16):
                observed = [lookup[key]["patch_texture_quality"][patch] for key in group
                            if not lookup[key]["objective_noop"][patch]]
                if len(observed) > 1:
                    monotonic_total += 1
                    monotonic_passed += int(all(a >= b - 1e-7 for a, b in zip(observed, observed[1:])))
    audit = {"status": "PASSED" if clean_ok and finite and supervised.any(axis=1).all() and
             monotonic_passed == monotonic_total else "FAILED",
             "counts": counts, "records": len(ordered), "assets": len({key[0] for key in ordered}),
             "clean_quality_one": clean_ok, "finite_quality": finite,
             "texture_supervision_patch_fraction": float(supervised.sum() / np.stack(
                 [lookup[key]["patch_mask"] for key in ordered]).sum()),
             "all_assets_have_texture_supervision": bool(supervised.any(axis=1).all()),
             "observable_monotonic_patch_sequences": {"passed": monotonic_passed,
                 "total": monotonic_total, "rate": monotonic_passed / max(monotonic_total, 1)}}
    metadata = {**shard_metadata[0], "scope": "formal", "records": len(ordered),
                "assets": audit["assets"], "composed_from_shards": len(args.shard_dirs),
                "shard_metadata": shard_metadata, "audit": audit}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
