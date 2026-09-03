#!/usr/bin/env python3
"""Compose Point/Mesh global and spatial texture features for full Train/Val quality learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TEXTURE_ATTACKS = {"texture_detail_loss", "texture_region_missing", "texture_misalignment"}


def key(row) -> tuple[str, str, str]:
    return str(row["asset_id"]), str(row["attack"]), str(row["level"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geometry-feature-dir", type=Path, required=True)
    parser.add_argument("--texture-audit", type=Path, nargs="+", required=True)
    parser.add_argument("--objective-target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audits = [json.loads(path.read_text(encoding="utf-8")) for path in args.texture_audit]
    if not all(item["passed"] for item in audits):
        raise ValueError("At least one texture feature shard failed")
    signatures = {item["signature"] for item in audits}
    if len(signatures) != 1:
        raise ValueError(f"Texture signature mismatch: {signatures}")
    texture_records = [row for item in audits for row in item["records"]]
    texture_lookup = {key(row): row["feature_path"] for row in texture_records}
    if len(texture_lookup) != len(texture_records):
        raise ValueError("Duplicate spatial texture feature keys")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val"):
        rows = [row for row in manifest["records"] if row["split"] == split]
        with np.load(args.geometry_feature_dir / f"features_{split}.npz") as values:
            geometry = {name: values[name].copy() for name in values.files}
        geometry_keys = list(zip(geometry["asset_ids"].astype(str), geometry["attacks"].astype(str),
                                 geometry["levels"].astype(str)))
        geometry_lookup = {item: index for index, item in enumerate(geometry_keys)}
        with np.load(args.objective_target_dir / f"objective_targets_v2_{split}.npz") as values:
            targets = {name: values[name].copy() for name in values.files}
        target_keys = list(zip(targets["asset_ids"].astype(str), targets["attacks"].astype(str),
                               targets["levels"].astype(str)))
        target_lookup = {item: index for index, item in enumerate(target_keys)}
        geometry_indices, target_indices = [], []
        texture_values = {name: [] for name in ("tokens", "token_mask", "view_stats", "asset_stats")}
        for row in rows:
            row_key = key(row)
            geometry_key = ((row_key[0], "clean", "clean")
                            if row["attack"] in TEXTURE_ATTACKS else row_key)
            if geometry_key not in geometry_lookup:
                raise KeyError(f"Missing geometry feature: {geometry_key}")
            if row_key not in texture_lookup:
                raise KeyError(f"Missing texture feature: {row_key}")
            geometry_indices.append(geometry_lookup[geometry_key])
            target_indices.append(target_lookup[row_key])
            with np.load(texture_lookup[row_key]) as values:
                for name in texture_values:
                    texture_values[name].append(values[name].copy())
        geometry_indices = np.asarray(geometry_indices, np.int64)
        target_indices = np.asarray(target_indices, np.int64)
        np.savez_compressed(
            args.output_dir / f"features_{split}.npz",
            asset_ids=np.asarray([row["asset_id"] for row in rows]),
            sheets=np.asarray([row["sheet"] for row in rows]),
            attacks=np.asarray([row["attack"] for row in rows]),
            levels=np.asarray([row["level"] for row in rows]),
            attack_index=np.asarray([row["attack_index"] for row in rows], np.int64),
            point_global=geometry["point_global"][geometry_indices],
            mesh_global=geometry["mesh_global"][geometry_indices],
            morphology=geometry["morphology"][geometry_indices],
            patches=geometry["patches"][geometry_indices],
            patch_mask=geometry["patch_mask"][geometry_indices],
            overall_quality=targets["overall_quality"][target_indices].astype(np.float32),
            geometry_quality=targets["geometry_quality"][target_indices].astype(np.float32),
            texture_quality=targets["texture_quality"][target_indices].astype(np.float32),
            objective_noop=targets["objective_noop"][target_indices],
            **{name: np.stack(values) for name, values in texture_values.items()},
        )
        counts[split] = len(rows)
    metadata = {"schema_version": 1, "status": "COMPLETE", "counts": counts,
                "texture_signature": signatures.pop(), "loaded_splits": ["train", "val"],
                "test_blind_loaded": False,
                "geometry_texture_attacks": "reuse same-asset clean geometry features",
                "texture_geometry_attacks": "extract attacked glTF visible texture features"}
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
