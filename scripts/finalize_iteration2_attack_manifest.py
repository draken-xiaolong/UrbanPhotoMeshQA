#!/usr/bin/env python3
"""Remove audited no-op and duplicate-severity records from Iteration 2."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


LEVEL_ORDER = {"light": 0, "medium": 1, "heavy": 2}


def key(row: dict) -> tuple[str, str, str]:
    return row["asset_id"], row["attack"], row["level"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    records = manifest["records"]
    by_key = {key(row): row for row in records}
    audit_by_key = {key(row): row for row in audit["records"]}
    excluded: dict[tuple[str, str, str], dict] = {}

    # A zero measured texture change is invalid supervision, irrespective of
    # its nominal attack parameters.
    for row_key, row in audit_by_key.items():
        if row.get("file_effect") is not None and float(row["file_effect"]) <= 0.0:
            excluded[row_key] = {**by_key[row_key], "exclusion_reason": "objective_texture_noop"}

    # Quantised low-face QEM outputs can make adjacent levels identical. Keep
    # the lower severity and remove only later records with the same digest.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["asset_id"], row["attack"])].append(row)
    for rows in groups.values():
        seen_digests = set()
        for row in sorted(rows, key=lambda value: LEVEL_ORDER[value["level"]]):
            row_key = key(row)
            if row["asset_digest"] in seen_digests:
                excluded[row_key] = {**row, "exclusion_reason": "duplicate_severity_asset_digest"}
            seen_digests.add(row["asset_digest"])

    kept = [row for row in records if key(row) not in excluded]
    counts = Counter(row["split"] for row in kept)
    attack_counts = Counter(row["attack"] for row in kept)
    output = {
        **{name: value for name, value in manifest.items() if name != "records"},
        "schema_version": 2,
        "quality_control": {
            "source_records": len(records),
            "formal_records": len(kept),
            "excluded": len(excluded),
            "rule": "exclude measured texture no-ops and only later byte-identical severity packages",
            "audit": str(args.audit),
        },
        "counts": dict(counts),
        "attack_counts": dict(attack_counts),
        "records": kept,
    }
    exclusions = {
        "schema_version": 1,
        "count": len(excluded),
        "reason_counts": dict(Counter(row["exclusion_reason"] for row in excluded.values())),
        "records": list(excluded.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.exclusions.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.exclusions.write_text(json.dumps(exclusions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"formal_records": len(kept), "counts": counts,
                      "attack_counts": attack_counts,
                      "reason_counts": exclusions["reason_counts"]}, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
