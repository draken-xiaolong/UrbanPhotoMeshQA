#!/usr/bin/env python3
"""Audit formal ordering, bounds and semantic invariants of OQI v2 targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GEOMETRY_ATTACKS = {
    "geometry_hole", "mesh_simplification_qem", "geometry_noise_spike",
}
TEXTURE_ATTACKS = {
    "texture_detail_loss", "texture_region_missing", "texture_misalignment",
}
QUALITY_ARRAYS = (
    "geometry_fidelity", "completeness", "topology_health", "texture_fidelity_raw",
    "geometry_quality_raw", "overall_quality_raw", "geometry_quality",
    "texture_quality", "overall_quality",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))["records"]
    order_checks = {}
    rows = []
    for split in ("train", "val", "test", "blind"):
        expected = [row for row in manifest if row["split"] == split]
        with np.load(args.target_dir / f"objective_targets_v2_{split}.npz") as values:
            keys = list(zip(
                values["asset_ids"].astype(str), values["attacks"].astype(str),
                values["levels"].astype(str),
            ))
            order_checks[split] = keys == [
                (row["asset_id"], row["attack"], row["level"]) for row in expected
            ]
            for index, (record, record_key) in enumerate(zip(expected, keys)):
                rows.append({
                    "record": record,
                    "key": record_key,
                    **{name: float(values[name][index]) for name in QUALITY_ARRAYS},
                    "objective_noop": bool(values["objective_noop"][index]),
                })

    quality = np.asarray([[row[name] for name in QUALITY_ARRAYS] for row in rows])
    finite = bool(np.isfinite(quality).all())
    bounded = bool(((quality >= -1e-7) & (quality <= 1.0 + 1e-7)).all())
    clean_valid = all(
        all(abs(row[name] - 1.0) <= 1e-7 for name in QUALITY_ARRAYS)
        and not row["objective_noop"]
        for row in rows if row["record"]["attack"] == "clean"
    )
    geometry_isolation = all(
        abs(row["texture_quality"] - 1.0) <= 1e-7
        for row in rows if row["record"]["attack"] in GEOMETRY_ATTACKS
    )
    texture_isolation = all(
        abs(row["geometry_quality"] - 1.0) <= 1e-7
        for row in rows if row["record"]["attack"] in TEXTURE_ATTACKS
    )
    noop_valid = all(
        abs(row["texture_quality"] - 1.0) <= 1e-7
        and abs(row["overall_quality"] - 1.0) <= 1e-7
        for row in rows if row["objective_noop"]
    )

    monotonicity = {}
    for attack in sorted(GEOMETRY_ATTACKS | TEXTURE_ATTACKS):
        groups = {}
        for row in rows:
            if row["record"]["attack"] == attack and not row["objective_noop"]:
                groups.setdefault(row["record"]["asset_id"], {})[
                    row["record"]["level"]
                ] = row["overall_quality"]
        passed = 0
        total = 0
        for values in groups.values():
            sequence = [values[level] for level in ("light", "medium", "heavy")
                        if level in values]
            passed += int(all(left >= right - 1e-7
                              for left, right in zip(sequence, sequence[1:])))
            total += 1
        monotonicity[attack] = {
            "passed": passed, "total": total, "rate": passed / max(total, 1)
        }

    monotonic_valid = all(value["passed"] == value["total"]
                          for value in monotonicity.values())
    passed = all((finite, bounded, clean_valid, geometry_isolation,
                  texture_isolation, noop_valid, monotonic_valid,
                  *order_checks.values()))
    report = {
        "schema_version": 2,
        "status": "PASSED" if passed else "FAILED",
        "records": len(rows),
        "all_finite": finite,
        "all_quality_values_bounded_0_1": bounded,
        "split_order_checks": order_checks,
        "clean_targets_all_one": clean_valid,
        "geometry_attacks_leave_texture_quality_one": geometry_isolation,
        "texture_attacks_leave_geometry_quality_one": texture_isolation,
        "objective_noops_remain_quality_one": noop_valid,
        "observable_monotonicity_passed": monotonic_valid,
        "objective_noop_records": int(sum(row["objective_noop"] for row in rows)),
        "observable_monotonicity_after_envelope": monotonicity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
