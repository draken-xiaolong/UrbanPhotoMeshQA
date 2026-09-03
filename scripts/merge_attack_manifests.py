#!/usr/bin/env python3
"""Merge disjoint split exporters and reject missing/duplicate attack records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=3312)
    args = parser.parse_args()
    records, seen = [], set()
    for path in args.inputs:
        for row in json.loads(path.read_text(encoding="utf-8"))["records"]:
            key = row["asset_id"], row["attack"], row["level"]
            if key in seen:
                raise ValueError(f"Duplicate record: {key}")
            seen.add(key); records.append(row)
    if len(records) != args.expected:
        raise ValueError(f"Expected {args.expected} records, got {len(records)}")
    payload = {"schema_version": 1, "seed": 2026, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
