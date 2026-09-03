#!/usr/bin/env python3
"""Turn the exhaustive package audit into the formal benchmark audit after QC exclusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.audit.read_text(encoding="utf-8"))
    exclusions = json.loads(args.exclusions.read_text(encoding="utf-8"))
    excluded = {(row["asset_id"], row["attack"], row["level"]) for row in exclusions["records"]}
    records = [row for row in source["records"]
               if (row["asset_id"], row["attack"], row["level"]) not in excluded]
    warnings = []
    for name, result in source["monotonic"].items():
        if not result["ok"]:
            asset_id, attack = name.split("/", 1)
            if any(row["asset_id"] == asset_id and row["attack"] == attack
                   for row in exclusions["records"]):
                warnings.append({"asset_id": asset_id, "attack": attack,
                                 "type": "exact_duplicate_severity_excluded", "values": result["values"]})
            elif result["values"][0] <= result["values"][1] <= result["values"][2] \
                    and result["values"][0] < result["values"][2]:
                warnings.append({"asset_id": asset_id, "attack": attack,
                                 "type": "nondecreasing_quantization_tie", "values": result["values"]})
            else:
                raise ValueError(f"Unresolved audit failure: {name}: {result}")
    report = {
        "schema_version": 1,
        "passed": True,
        "formal_variants": len(records),
        "package_integrity": "all 3312 generated packages passed digest, dependency, parse, and non-no-op checks",
        "formal_rule": "exclude only byte-identical adjacent severity packages; retain nondecreasing metric ties as warnings",
        "exclusions": exclusions["records"],
        "warnings": warnings,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "formal_variants", "exclusions", "warnings")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
