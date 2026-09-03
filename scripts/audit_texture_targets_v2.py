#!/usr/bin/env python3
"""Audit ordering, masks, no-op labels, monotonicity and repeatability of texture v2 targets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


TEXTURE_ATTACKS = {
    "texture_detail_loss",
    "texture_region_missing",
    "texture_misalignment",
}
PRIMARY = {
    "texture_detail_loss": "masked_gradient_mae",
    "texture_region_missing": "surface_changed_fraction_1_255",
    "texture_misalignment": "surface_rgb_rmse",
}


def load_targets(root: Path):
    rows = {}
    split_keys = {}
    names = None
    for split in ("train", "val", "test", "blind"):
        with np.load(root / f"texture_targets_v2_{split}.npz") as values:
            current = values["metric_names"].astype(str).tolist()
            if names is None:
                names = current
            elif names != current:
                raise ValueError(f"Metric names differ in {split}")
            keys = list(zip(
                values["asset_ids"].astype(str),
                values["attacks"].astype(str),
                values["levels"].astype(str),
            ))
            split_keys[split] = keys
            for index, key in enumerate(keys):
                rows[key] = {
                    "metrics": values["metrics"][index].copy(),
                    "common_textured_pixels": int(values["common_textured_pixels"][index]),
                    "textured_foreground_fraction": float(values["textured_foreground_fraction"][index]),
                    "maximum_rgb_difference": int(values["maximum_rgb_difference"][index]),
                    "objective_noop": bool(values["objective_noop"][index]),
                    "split": split,
                }
    return names, rows, split_keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--repeat-target-dir", type=Path)
    parser.add_argument("--compare-target-dir", type=Path)
    parser.add_argument("--repeat-tolerance", type=float, default=1e-7)
    parser.add_argument("--asset-ids", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names, rows, split_keys = load_targets(args.target_dir)
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))["records"]
    requested = set(args.asset_ids or [])
    order_checks = {}
    for split in ("train", "val", "test", "blind"):
        expected = [
            (row["asset_id"], row["attack"], row["level"])
            for row in manifest
            if row["split"] == split and row["attack"] in TEXTURE_ATTACKS | {"clean"}
            and (not requested or row["asset_id"] in requested)
        ]
        order_checks[split] = expected == split_keys[split]

    metric_index = {name: index for index, name in enumerate(names)}
    all_metrics = np.stack([row["metrics"] for row in rows.values()])
    masks_valid = all(
        row["common_textured_pixels"] > 0
        and 0.0 <= row["textured_foreground_fraction"] <= 1.0
        for row in rows.values()
    )
    noops = Counter(
        (key[1], key[2]) for key, row in rows.items()
        if key[1] != "clean" and row["objective_noop"]
    )
    noops_by_attack_level = {
        attack: {level: noops[(attack, level)] for level in ("light", "medium", "heavy")}
        for attack in sorted(TEXTURE_ATTACKS)
    }

    monotonicity = {}
    failures = []
    assets = sorted({key[0] for key in rows})
    for attack, metric_name in PRIMARY.items():
        index = metric_index[metric_name]
        passed = 0
        total = 0
        for asset_id in assets:
            keys = [(asset_id, attack, level) for level in ("light", "medium", "heavy")]
            if not all(key in rows for key in keys):
                continue
            values = [float(rows[key]["metrics"][index]) for key in keys]
            ok = all(left <= right + 1e-12 for left, right in zip(values, values[1:]))
            total += 1
            passed += int(ok)
            if not ok:
                failures.append({"asset_id": asset_id, "attack": attack,
                                 "metric": metric_name, "values": values})
        monotonicity[attack] = {"metric": metric_name, "passed": passed,
                                "total": total, "rate": passed / max(total, 1)}

    repeatability = None
    if args.repeat_target_dir:
        _, repeated, _ = load_targets(args.repeat_target_dir)
        common = sorted(set(rows) & set(repeated))
        maximum = max((float(np.max(np.abs(rows[key]["metrics"]
                                             - repeated[key]["metrics"])))
                       for key in common), default=0.0)
        flags_equal = all(rows[key]["objective_noop"] == repeated[key]["objective_noop"]
                          for key in common)
        repeatability = {
            "records": len(common),
            "maximum_absolute_metric_difference": maximum,
            "noop_flags_equal": flags_equal,
            "tolerance": args.repeat_tolerance,
            "passed": maximum <= args.repeat_tolerance and flags_equal,
        }

    resolution_comparison = None
    if args.compare_target_dir:
        compare_names, compared, _ = load_targets(args.compare_target_dir)
        if compare_names != names:
            raise ValueError("Comparison metric names differ")
        common = sorted(key for key in set(rows) & set(compared) if key[1] != "clean")
        correlations = {}
        for index, name in enumerate(names):
            left = [rows[key]["metrics"][index] for key in common]
            right = [compared[key]["metrics"][index] for key in common]
            correlations[name] = float(spearmanr(left, right).statistic)
        resolution_comparison = {
            "records": len(common),
            "metric_spearman": correlations,
            "target_noops": int(sum(rows[key]["objective_noop"] for key in common)),
            "comparison_noops": int(sum(compared[key]["objective_noop"] for key in common)),
        }

    passed = (
        np.isfinite(all_metrics).all()
        and all(order_checks.values())
        and masks_valid
        and (repeatability is None or repeatability["passed"])
    )
    report = {
        "schema_version": 2,
        "status": "PASSED" if passed else "FAILED",
        "records": len(rows),
        "all_finite": bool(np.isfinite(all_metrics).all()),
        "split_order_checks": order_checks,
        "visible_mask_checks_passed": masks_valid,
        "minimum_common_textured_pixels": min(
            row["common_textured_pixels"] for row in rows.values()
        ),
        "objective_noops_by_attack_level": noops_by_attack_level,
        "raw_primary_metric_monotonicity": monotonicity,
        "raw_monotonicity_failures": failures,
        "repeatability": repeatability,
        "resolution_comparison": resolution_comparison,
        "monotonicity_policy": (
            "Raw metrics and objective no-op flags remain unmodified; scalar quality "
            "may apply a cumulative severity envelope within each asset and attack."
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
