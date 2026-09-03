#!/usr/bin/env python3
"""Extract deterministic local face-patch descriptors for no-reference mesh QA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from urbanphotomeshqa.gltf import GltfReader  # noqa: E402
from urbanphotomeshqa.mesh_attacks import apply_mesh_attack  # noqa: E402
from urbanphotomeshqa.patches import patch_descriptors  # noqa: E402


GEOMETRY_ATTACKS = {"connected_crop", "hole", "retriangulate"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patches", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = {row["asset_id"]: row for row in manifest["records"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"seed": args.seed, "patches": args.patches, "neighbors": args.neighbors,
                "descriptor_dim": 58, "geometry_attacks": sorted(GEOMETRY_ATTACKS), "splits": {}}
    for split in ("train", "val", "test", "blind"):
        with np.load(args.feature_dir / f"scores_{split}.npz") as values:
            query_ids = values["query_ids"].astype(str)
            attacks = values["attacks"].astype(str)
            severities = values["severities"].astype(np.float32)
            targets = values["targets"].astype(np.int64)
            gallery_count = len(values["gallery_point"])
        gallery_ids = np.empty(gallery_count, dtype=object)
        for gallery_index in range(gallery_count):
            gallery_ids[gallery_index] = query_ids[np.flatnonzero(targets == gallery_index)[0]]
        gallery_patch = np.zeros((gallery_count, args.patches, 58), np.float32)
        gallery_mask = np.zeros((gallery_count, args.patches), bool)
        query_patch = np.zeros((len(query_ids), args.patches, 58), np.float32)
        query_mask = np.zeros((len(query_ids), args.patches), bool)
        for asset_index, asset_id in enumerate(gallery_ids.tolist()):
            row = records[str(asset_id)]
            clean = GltfReader(PROJECT / row["gltf_path"]).load_mesh()
            descriptor, mask = patch_descriptors(clean, args.patches, args.neighbors)
            gallery_patch[asset_index], gallery_mask[asset_index] = descriptor, mask
            query_indices = np.flatnonzero(query_ids == asset_id)
            query_patch[query_indices] = descriptor
            query_mask[query_indices] = mask
            for query_index in query_indices.tolist():
                attack = attacks[query_index]
                if attack not in GEOMETRY_ATTACKS:
                    continue
                attacked = apply_mesh_attack(
                    clean, attack, float(severities[query_index]),
                    args.seed + asset_index * 1009 + query_index,
                )
                query_patch[query_index], query_mask[query_index] = patch_descriptors(
                    attacked, args.patches, args.neighbors)
            if asset_index == 0 or (asset_index + 1) % 8 == 0 or asset_index + 1 == gallery_count:
                print(f"{split}: {asset_index + 1}/{gallery_count}", flush=True)
        np.savez_compressed(
            args.output_dir / f"patches_{split}.npz",
            gallery_ids=gallery_ids.astype(str), query_ids=query_ids,
            gallery_patch=gallery_patch, gallery_mask=gallery_mask,
            query_patch=query_patch, query_mask=query_mask,
        )
        metadata["splits"][split] = {"gallery": gallery_count, "queries": len(query_ids)}
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PATCH_FEATURES_COMPLETE", **metadata}, indent=2))


if __name__ == "__main__":
    main()
