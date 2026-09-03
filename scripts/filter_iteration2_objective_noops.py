#!/usr/bin/env python3
"""Remove UV-invisible objective no-ops without recomputing target metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def keys(values) -> list[tuple[str, str, str]]:
    return list(zip(values["asset_ids"].astype(str), values["attacks"].astype(str),
                    values["levels"].astype(str)))


def filter_npz(source: Path, target: Path, excluded: set[tuple[str, str, str]]) -> int:
    with np.load(source) as values:
        arrays = {name: values[name].copy() for name in values.files}
    keep = np.asarray([item not in excluded for item in keys(arrays)], dtype=bool)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{
        name: value[keep] if value.ndim and len(value) == len(keep) else value
        for name, value in arrays.items()
    })
    return int(keep.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--objective-dir", type=Path, required=True)
    parser.add_argument("--texture-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-objective-dir", type=Path, required=True)
    parser.add_argument("--output-texture-dir", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    args = parser.parse_args()

    excluded = set()
    rows = []
    for split in ("train", "val", "test", "blind"):
        with np.load(args.objective_dir / f"objective_targets_v2_{split}.npz") as values:
            for item, noop in zip(keys(values), values["objective_noop"]):
                if bool(noop):
                    excluded.add(item)
                    rows.append({"asset_id": item[0], "attack": item[1], "level": item[2],
                                 "split": split, "exclusion_reason": "uv_invisible_objective_noop"})

    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    dataset["records"] = [row for row in dataset["records"]
                          if (row["asset_id"], row["attack"], row["level"]) not in excluded]
    dataset["schema_version"] = 2
    dataset["quality_control"] = {
        **dataset.get("quality_control", {}),
        "excluded_uv_invisible_objective_noops": len(excluded),
        "objective_noop_exclusions": str(args.exclusions),
    }
    counts = {}
    for split in ("train", "val", "test", "blind"):
        subset = [row for row in dataset["records"] if row["split"] == split]
        counts[split] = {"records": len(subset), "assets": len({row["asset_id"] for row in subset})}
        filter_npz(args.objective_dir / f"objective_targets_v2_{split}.npz",
                   args.output_objective_dir / f"objective_targets_v2_{split}.npz", excluded)
        filter_npz(args.texture_dir / f"texture_targets_v2_{split}.npz",
                   args.output_texture_dir / f"texture_targets_v2_{split}.npz", excluded)
    dataset["splits"] = counts
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = {"schema_version": 1, "count": len(rows),
               "attack_counts": dict(Counter(row["attack"] for row in rows)), "records": rows}
    args.exclusions.parent.mkdir(parents=True, exist_ok=True)
    args.exclusions.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {"schema_version": 2, "filtered_from": str(args.objective_dir),
                "excluded": len(rows), "counts": counts}
    for root in (args.output_objective_dir, args.output_texture_dir):
        (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"excluded": len(rows), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
