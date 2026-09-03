#!/usr/bin/env python3
"""Combine clean official packages and exported attacks into the no-reference QA protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ATTACKS = ("clean", "geometry_hole", "mesh_simplification_qem", "geometry_noise_spike",
           "texture_detail_loss", "texture_region_missing", "texture_misalignment")
LEVEL_SEVERITY = {"clean": 0.0, "light": 1.0 / 3.0, "medium": 2.0 / 3.0, "heavy": 1.0}
GEOMETRY = {"geometry_hole", "mesh_simplification_qem", "geometry_noise_spike"}
TEXTURE = {"texture_detail_loss", "texture_region_missing", "texture_misalignment"}


def targets(attack: str, level: str) -> dict:
    severity = LEVEL_SEVERITY[level]
    return {
        "attack_index": ATTACKS.index(attack), "severity": severity,
        "overall_quality": 1.0 - severity,
        "geometry_quality": 1.0 - severity if attack in GEOMETRY else 1.0,
        "texture_quality": 1.0 - severity if attack in TEXTURE else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--attack-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))["records"]
    attacked = json.loads(args.attack_manifest.read_text(encoding="utf-8"))["records"]
    records = []
    for record in source:
        gltf = (args.source_root / record["sheet"] / record.get("class_name", "BUILDING")
                / record["asset_id"] / f"{record['asset_id']}.gltf")
        records.append({"asset_id": record["asset_id"], "sheet": record["sheet"],
                        "class_name": "BUILDING", "split": record["split"], "attack": "clean",
                        "level": "clean", "parameters": {}, "gltf_path": str(gltf),
                        **targets("clean", "clean")})
    for record in attacked:
        records.append({**record, **targets(record["attack"], record["level"])})
    counts = {}
    for split in ("train", "val", "test", "blind"):
        subset = [row for row in records if row["split"] == split]
        counts[split] = {"records": len(subset), "assets": len({row["asset_id"] for row in subset})}
    payload = {"schema_version": 1, "seed": 2026, "attacks": list(ATTACKS),
               "severity": LEVEL_SEVERITY, "splits": counts, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "splits": counts}))


if __name__ == "__main__":
    main()
