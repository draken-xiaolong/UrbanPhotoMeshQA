#!/usr/bin/env python3
"""Compose disjoint attack-generation manifest shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def key(row):
    return row["asset_id"], row["attack"], row["level"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    signatures = {payload.get("generator_signature") for payload in payloads}
    if len(signatures) != 1:
        raise ValueError(f"Generator signatures differ: {signatures}")
    records = {}
    for payload in payloads:
        for row in payload["records"]:
            if key(row) in records:
                raise ValueError(f"Duplicate attack record: {key(row)}")
            records[key(row)] = row
    ordered = [records[item] for item in sorted(records)]
    output = {
        "schema_version": 2,
        "seed": payloads[0]["seed"],
        "generator_signature": signatures.pop(),
        "composed_from": [str(path) for path in args.inputs],
        "records": ordered,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETE", "records": len(ordered)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
