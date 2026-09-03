#!/usr/bin/env python3
"""Audit ordering, finiteness, repeatability and raw monotonicity of geometry v2 targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GEOMETRY_ATTACKS = {
    "geometry_hole",
    "mesh_simplification_qem",
    "geometry_noise_spike",
}
PRIMARY = {
    "geometry_hole": "clean_missing_fraction_0p005",
    "mesh_simplification_qem": "symmetric_chamfer_l1",
    "geometry_noise_spike": "symmetric_chamfer_l1",
}


def load_targets(root: Path):
    output = {}
    split_keys = {}
    names = None
    finite = True
    for split in ("train", "val", "test", "blind"):
        with np.load(root / f"geometry_targets_v2_{split}.npz") as values:
            current_names = values["metric_names"].astype(str).tolist()
            if names is None:
                names = current_names
            elif names != current_names:
                raise ValueError(f"Metric names differ in {split}")
            keys = list(zip(
                values["asset_ids"].astype(str),
                values["attacks"].astype(str),
                values["levels"].astype(str),
            ))
            split_keys[split] = keys
            finite &= bool(np.isfinite(values["metrics"]).all())
            output.update(zip(keys, values["metrics"].copy()))
    return names, output, split_keys, finite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--repeat-target-dir", type=Path)
    parser.add_argument("--repeat-tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names, targets, split_keys, finite = load_targets(args.target_dir)
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))["records"]
    order_checks = {}
    for split in ("train", "val", "test", "blind"):
        expected = [
            (row["asset_id"], row["attack"], row["level"])
            for row in manifest
            if row["split"] == split
            and row["attack"] in GEOMETRY_ATTACKS | {"clean"}
        ]
        order_checks[split] = expected == split_keys[split]

    metric_index = {name: index for index, name in enumerate(names)}
    monotonic = {}
    failures = []
    asset_ids = sorted({key[0] for key in targets})
    for attack, metric_name in PRIMARY.items():
        passed = 0
        total = 0
        for asset_id in asset_ids:
            keys = [
                (asset_id, attack, level)
                for level in ("light", "medium", "heavy")
                if (asset_id, attack, level) in targets
            ]
            values = [
                float(targets[key][metric_index[metric_name]]) for key in keys
            ]
            ok = all(left <= right + 1e-12
                     for left, right in zip(values, values[1:]))
            total += 1
            passed += int(ok)
            if not ok:
                failures.append({
                    "asset_id": asset_id,
                    "attack": attack,
                    "metric": metric_name,
                    "values": values,
                })
        monotonic[attack] = {
            "passed": passed,
            "total": total,
            "rate": passed / max(total, 1),
        }

    repeat = None
    if args.repeat_target_dir:
        _, repeated, _, repeated_finite = load_targets(args.repeat_target_dir)
        common = sorted(set(targets) & set(repeated))
        maximum = max(
            (float(np.max(np.abs(targets[key] - repeated[key]))) for key in common),
            default=0.0,
        )
        repeat = {
            "records": len(common),
            "finite": repeated_finite,
            "maximum_absolute_difference": maximum,
            "tolerance": args.repeat_tolerance,
            "passed": repeated_finite and maximum <= args.repeat_tolerance,
        }

    passed = finite and all(order_checks.values()) and (
        repeat is None or repeat["passed"]
    )
    report = {
        "schema_version": 2,
        "status": "PASSED" if passed else "FAILED",
        "records": len(targets),
        "all_finite": finite,
        "split_order_checks": order_checks,
        "repeatability": repeat,
        "raw_primary_metric_monotonicity": monotonic,
        "raw_monotonicity_failures": failures,
        "monotonicity_policy": (
            "Raw metrics remain unmodified; scalar quality applies a cumulative "
            "severity envelope within each asset and attack."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items()
                      if key != "raw_monotonicity_failures"}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
