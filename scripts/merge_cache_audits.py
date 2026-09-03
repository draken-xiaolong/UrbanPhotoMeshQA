#!/usr/bin/env python3
"""Merge disjoint cache audits in the locked quality-manifest order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def key(row):
    return row["asset_id"], row["attack"], row["level"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, signatures = {}, set()
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("passed", False):
            raise ValueError(f"Cache audit did not pass: {path}")
        signatures.add(payload["extractor_signature"])
        for row in payload["records"]:
            item_key = key(row)
            if item_key in records:
                raise ValueError(f"Duplicate cache record: {item_key}")
            records[item_key] = row
    if len(signatures) != 1:
        raise ValueError(f"Extractor signature mismatch: {signatures}")
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))["records"]
    ordered, missing = [], []
    for row in dataset:
        item = records.get(key(row))
        if item is None:
            missing.append(key(row))
        else:
            ordered.append(item)
    if missing or len(records) != len(dataset):
        raise ValueError(f"Expected {len(dataset)} unique records; got {len(records)}; missing={missing[:10]}")
    payload = {"schema_version": 1, "passed": True, "errors": [],
               "extractor_signature": signatures.pop(), "records": ordered}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "records": len(ordered), "output": str(args.output)}))


if __name__ == "__main__":
    main()
