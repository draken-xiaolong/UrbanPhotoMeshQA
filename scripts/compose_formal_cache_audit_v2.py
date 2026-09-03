#!/usr/bin/env python3
"""Combine reused clean/texture caches with newly extracted geometry v2 caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GEOMETRY_ATTACKS = {
    "geometry_hole",
    "mesh_simplification_qem",
    "geometry_noise_spike",
}


def key(row):
    return row["asset_id"], row["attack"], row["level"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--reused-audit", type=Path, required=True)
    parser.add_argument("--geometry-audits", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reused_payload = json.loads(args.reused_audit.read_text(encoding="utf-8"))
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.geometry_audits
    ]
    if not reused_payload.get("passed", False) or not all(
        payload.get("passed", False) for payload in payloads
    ):
        raise ValueError("Every input cache audit must pass")
    signatures = {
        reused_payload["extractor_signature"],
        *(payload["extractor_signature"] for payload in payloads),
    }
    if len(signatures) != 1:
        raise ValueError(f"Extractor signatures differ: {signatures}")

    reused = {
        key(row): row for row in reused_payload["records"]
        if row["attack"] not in GEOMETRY_ATTACKS
    }
    geometry = {}
    for payload in payloads:
        for row in payload["records"]:
            if row["attack"] not in GEOMETRY_ATTACKS:
                raise ValueError(f"Unexpected non-geometry v2 cache: {key(row)}")
            if key(row) in geometry:
                raise ValueError(f"Duplicate geometry v2 cache: {key(row)}")
            geometry[key(row)] = row

    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))["records"]
    ordered = []
    missing = []
    for row in dataset:
        source = geometry if row["attack"] in GEOMETRY_ATTACKS else reused
        cached = source.get(key(row))
        if cached is None:
            missing.append(key(row))
        else:
            ordered.append(cached)
    if missing or len(ordered) != len(dataset):
        raise ValueError(f"Missing caches: {missing[:10]}; got {len(ordered)}/{len(dataset)}")
    if len(reused) + len(geometry) != len(dataset):
        raise ValueError(
            f"Unexpected cache count: reused={len(reused)}, geometry={len(geometry)}, "
            f"dataset={len(dataset)}"
        )

    report = {
        "schema_version": 2,
        "passed": True,
        "errors": [],
        "extractor_signature": signatures.pop(),
        "records": ordered,
        "lineage": {
            "dataset_manifest": str(args.dataset_manifest),
            "reused_audit": str(args.reused_audit),
            "geometry_audits": [str(path) for path in args.geometry_audits],
            "reused_clean_and_texture_records": len(reused),
            "new_geometry_records": len(geometry),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "COMPLETE",
        "records": len(ordered),
        "reused_clean_and_texture_records": len(reused),
        "new_geometry_records": len(geometry),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
