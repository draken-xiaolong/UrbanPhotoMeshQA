#!/usr/bin/env python3
"""Create a formal v2 dataset manifest by replacing only geometry records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GEOMETRY_ATTACKS = {
    "geometry_hole",
    "mesh_simplification_qem",
    "geometry_noise_spike",
}
FORMAL_COUNTS = {"train": 1518, "val": 760, "test": 607, "blind": 608}


def key(row):
    return row["asset_id"], row["attack"], row["level"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-v1", type=Path, required=True)
    parser.add_argument("--geometry-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.formal_v1.read_text(encoding="utf-8"))
    geometry = json.loads(args.geometry_v2.read_text(encoding="utf-8"))
    replacements = {key(row): row for row in geometry["records"]}
    records = []
    replaced = 0
    for old in base["records"]:
        if old["attack"] not in GEOMETRY_ATTACKS:
            records.append(old)
            continue
        new = replacements.get(key(old))
        if new is None:
            raise ValueError(f"Missing v2 geometry record: {key(old)}")
        # Preserve the formal split, ordered sample identity and target fields;
        # replace all file-generation provenance with the v2 record.
        records.append({
            **new,
            "attack_index": old["attack_index"],
            "severity": old["severity"],
            "overall_quality": old["overall_quality"],
            "geometry_quality": old["geometry_quality"],
            "texture_quality": old["texture_quality"],
        })
        replaced += 1

    counts = {
        split: sum(row["split"] == split for row in records)
        for split in FORMAL_COUNTS
    }
    if counts != FORMAL_COUNTS:
        raise ValueError(f"Formal counts changed: {counts}")
    expected_replacements = sum(
        row["attack"] in GEOMETRY_ATTACKS for row in base["records"]
    )
    if replaced != expected_replacements:
        raise ValueError(f"Expected {expected_replacements} replacements, got {replaced}")

    output = {
        **base,
        "schema_version": 2,
        "records": records,
        "geometry_data_version": "precision-safe local-coordinate export v2",
        "geometry_generator_signature": geometry.get("generator_signature"),
        "lineage": {
            "formal_v1": str(args.formal_v1),
            "geometry_v2": str(args.geometry_v2),
            "geometry_records_replaced": replaced,
            "clean_and_texture_records_reused": len(records) - replaced,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "COMPLETE",
        "records": len(records),
        "counts": counts,
        "geometry_records_replaced": replaced,
        "clean_and_texture_records_reused": len(records) - replaced,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
