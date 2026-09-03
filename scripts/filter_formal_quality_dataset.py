#!/usr/bin/env python3
"""Exclude exact duplicate severity packages from the formal QA benchmark."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def key(asset_id, attack, level):
    return str(asset_id), str(attack), str(level)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--objective-target-dir", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-cache-audit", type=Path, required=True)
    parser.add_argument("--output-feature-dir", type=Path, required=True)
    parser.add_argument("--output-objective-target-dir", type=Path, required=True)
    args = parser.parse_args()
    exclusions_payload = json.loads(args.exclusions.read_text(encoding="utf-8"))
    excluded = {key(row["asset_id"], row["attack"], row["level"])
                for row in exclusions_payload["records"]}

    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    original_count = len(dataset["records"])
    dataset["records"] = [row for row in dataset["records"]
                          if key(row["asset_id"], row["attack"], row["level"]) not in excluded]
    dataset["quality_control"] = {
        "excluded_exact_duplicate_severity_packages": len(excluded),
        "exclusions_file": str(args.exclusions),
        "rule": "remove only records whose adjacent severity glTF asset digests are byte-identical",
    }
    for split in ("train", "val", "test", "blind"):
        rows = [row for row in dataset["records"] if row["split"] == split]
        dataset["splits"][split] = {"records": len(rows), "assets": len({row["asset_id"] for row in rows})}
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = json.loads(args.cache_audit.read_text(encoding="utf-8"))
    audit["records"] = [row for row in audit["records"]
                        if key(row["asset_id"], row["attack"], row["level"]) not in excluded]
    audit["formal_exclusions"] = exclusions_payload
    audit["passed"] = bool(audit.get("passed", True))
    args.output_cache_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_cache_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    args.output_feature_dir.mkdir(parents=True, exist_ok=True)
    args.output_objective_target_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test", "blind"):
        feature_path = args.feature_dir / f"features_{split}.npz"
        with np.load(feature_path) as values:
            arrays = {name: values[name].copy() for name in values.files}
        keep = np.asarray([key(a, b, c) not in excluded for a, b, c in zip(
            arrays["asset_ids"], arrays["attacks"], arrays["levels"])])
        np.savez_compressed(args.output_feature_dir / feature_path.name,
                            **{name: value[keep] for name, value in arrays.items()})

        objective_path = args.objective_target_dir / f"objective_targets_{split}.npz"
        with np.load(objective_path) as values:
            arrays = {name: values[name].copy() for name in values.files}
        objective_keep = np.asarray([key(a, b, c) not in excluded for a, b, c in zip(
            arrays["asset_ids"], arrays["attacks"], arrays["levels"])])
        np.savez_compressed(args.output_objective_target_dir / objective_path.name,
                            **{name: value[objective_keep] for name, value in arrays.items()})
    for source, destination in (
        (args.feature_dir / "metadata.json", args.output_feature_dir / "metadata_source.json"),
        (args.objective_target_dir / "metadata.json", args.output_objective_target_dir / "metadata_source.json"),
    ):
        if source.is_file():
            shutil.copy2(source, destination)
    metadata = {"schema_version": 1, "original_records": original_count,
                "formal_records": len(dataset["records"]), "excluded": len(excluded),
                "exclusions": exclusions_payload["records"], "splits": dataset["splits"]}
    for root in (args.output_feature_dir, args.output_objective_target_dir):
        (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
