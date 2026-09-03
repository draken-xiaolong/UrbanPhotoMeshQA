#!/usr/bin/env python3
"""Replace selected formal texture records while preserving ordered sample identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TEXTURE_ATTACKS = {
    "texture_detail_loss", "texture_region_missing", "texture_misalignment",
}
FORMAL_COUNTS = {"train": 1518, "val": 760, "test": 607, "blind": 608}


def key(row):
    return row["asset_id"], row["attack"], row["level"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-base", type=Path, required=True)
    parser.add_argument("--texture-replacements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    base = json.loads(args.formal_base.read_text(encoding="utf-8"))
    replacement_payload = json.loads(args.texture_replacements.read_text(encoding="utf-8"))
    replacements = {
        key(row): row for row in replacement_payload["records"]
        if row["attack"] in TEXTURE_ATTACKS
    }
    unknown = set(replacements) - {key(row) for row in base["records"]}
    if unknown:
        raise ValueError(f"Replacement keys outside formal manifest: {sorted(unknown)[:3]}")
    records = []
    replaced = 0
    for old in base["records"]:
        new = replacements.get(key(old))
        if new is None:
            records.append(old)
            continue
        records.append({
            **new,
            "attack_index": old["attack_index"],
            "severity": old["severity"],
            "overall_quality": old["overall_quality"],
            "geometry_quality": old["geometry_quality"],
            "texture_quality": old["texture_quality"],
        })
        replaced += 1
    if replaced != len(replacements):
        raise ValueError(f"Expected {len(replacements)} replacements, got {replaced}")
    expected_full = sum(row["attack"] in TEXTURE_ATTACKS for row in base["records"])
    if not args.allow_partial and replaced != expected_full:
        raise ValueError(f"Full replacement requires {expected_full} records, got {replaced}")
    counts = {split: sum(row["split"] == split for row in records) for split in FORMAL_COUNTS}
    if counts != FORMAL_COUNTS:
        raise ValueError(f"Formal counts changed: {counts}")

    output = {
        **base,
        "schema_version": max(int(base.get("schema_version", 1)), 2),
        "records": records,
        "texture_data_version": "UV-usage-aware local texture attacks v2",
        "texture_generator_signature": replacement_payload.get("generator_signature"),
        "lineage": {
            **base.get("lineage", {}),
            "texture_replacements": str(args.texture_replacements),
            "texture_records_replaced": replaced,
            "texture_replacement_is_partial": replaced != expected_full,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETE", "records": len(records),
                      "replaced": replaced, "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
