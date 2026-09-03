#!/usr/bin/env python3
"""Catalog HK3D glTF packages and deterministically select Iteration 2 assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


MAIN_SPLITS = {"train": 210, "val": 45, "test": 45}


def canonical_json_sha(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inspect_asset(gltf: Path, source_root: Path) -> dict:
    asset_id = gltf.parent.name
    tile = gltf.parents[2].name
    record = {
        "asset_id": asset_id, "tile": tile,
        "source_gltf": str(gltf.relative_to(source_root)),
        "status": "qualified", "issues": [],
    }
    try:
        root = json.loads(gltf.read_text(encoding="utf-8"))
    except Exception as error:
        record.update(status="unusable", issues=[f"invalid_gltf_json:{type(error).__name__}"])
        return record
    dependencies = []
    for section in ("buffers", "images"):
        for item in root.get(section, []):
            uri = item.get("uri")
            if not uri or uri.startswith("data:"):
                continue
            path = gltf.parent / uri
            dependencies.append(path)
            if not path.is_file():
                record["issues"].append(f"missing_dependency:{uri}")
    accessors = root.get("accessors", [])
    mesh_primitives = [primitive for mesh in root.get("meshes", []) for primitive in mesh.get("primitives", [])]
    face_count = 0; vertex_count = 0; uv_primitives = 0
    for primitive in mesh_primitives:
        attributes = primitive.get("attributes", {})
        position = attributes.get("POSITION")
        if isinstance(position, int) and 0 <= position < len(accessors):
            vertex_count += int(accessors[position].get("count", 0))
        texcoord = attributes.get("TEXCOORD_0")
        if isinstance(texcoord, int) and 0 <= texcoord < len(accessors):
            uv_primitives += 1
        indices = primitive.get("indices")
        if isinstance(indices, int) and 0 <= indices < len(accessors):
            face_count += int(accessors[indices].get("count", 0)) // 3
        elif isinstance(position, int) and 0 <= position < len(accessors):
            face_count += int(accessors[position].get("count", 0)) // 3
    texture_pixels = 0; valid_images = 0
    for item in root.get("images", []):
        uri = item.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        path = gltf.parent / uri
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                texture_pixels += int(image.width * image.height)
                valid_images += 1
        except Exception as error:
            record["issues"].append(f"invalid_image:{uri}:{type(error).__name__}")
    if not mesh_primitives or face_count <= 0 or vertex_count <= 0:
        record["issues"].append("empty_mesh")
    if not root.get("buffers"):
        record["issues"].append("missing_buffer_declaration")
    if valid_images <= 0 or uv_primitives <= 0:
        record["issues"].append("texture_or_uv_unavailable")
    fatal = any(issue.startswith(("missing_dependency", "invalid_image", "empty_mesh", "invalid_gltf",
                                  "missing_buffer")) for issue in record["issues"])
    if fatal:
        record["status"] = "unusable"
    elif record["issues"]:
        record["status"] = "usable_natural_defect"
    record.update(
        face_count=face_count, vertex_count=vertex_count, texture_pixels=texture_pixels,
        image_count=valid_images, primitive_count=len(mesh_primitives), uv_primitive_count=uv_primitives,
        dependency_count=len(dependencies), gltf_canonical_sha256=canonical_json_sha(gltf),
    )
    return record


def diversity_order(records: list[dict], seed: int) -> list[dict]:
    if not records:
        return []
    faces = np.log1p([record["face_count"] for record in records])
    textures = np.log1p([record["texture_pixels"] for record in records])
    face_edges = np.quantile(faces, [0.2, 0.4, 0.6, 0.8])
    texture_edges = np.quantile(textures, [0.25, 0.5, 0.75])
    buckets: dict[tuple, list[dict]] = {}
    for record, face, texture in zip(records, faces, textures):
        bucket = (record["tile"], int(np.searchsorted(face_edges, face)),
                  int(np.searchsorted(texture_edges, texture)), record["status"])
        buckets.setdefault(bucket, []).append(record)
    for bucket, values in buckets.items():
        values.sort(key=lambda record: hashlib.sha256(
            f"{seed}|{bucket}|{record['asset_id']}".encode()).hexdigest())
    ordered = []
    keys = sorted(buckets)
    while keys:
        remaining = []
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    return ordered


def assign_main_splits(records: list[dict], seed: int) -> list[dict]:
    # Greedily match global quotas and each tile's expected split proportions.
    ordered = diversity_order(records, seed)
    tile_totals = {tile: sum(record["tile"] == tile for record in records)
                   for tile in {record["tile"] for record in records}}
    remaining = dict(MAIN_SPLITS)
    assigned = {split: {tile: 0 for tile in tile_totals} for split in MAIN_SPLITS}
    output = []
    total = len(records)
    for record in ordered:
        candidates = []
        for split, quota in MAIN_SPLITS.items():
            if remaining[split] <= 0:
                continue
            expected_tile = tile_totals[record["tile"]] * quota / total
            tile_fill = assigned[split][record["tile"]] / max(expected_tile, 1e-8)
            global_fill = (quota - remaining[split]) / quota
            tie = hashlib.sha256(f"{seed}|{record['asset_id']}|{split}".encode()).hexdigest()
            candidates.append((tile_fill + 0.25 * global_fill, tie, split))
        split = min(candidates)[2]
        output.append({**record, "split": split})
        remaining[split] -= 1
        assigned[split][record["tile"]] += 1
    if any(remaining.values()):
        raise RuntimeError(f"Split quota assignment failed: {remaining}")
    return sorted(output, key=lambda record: (record["split"], record["tile"], record["asset_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--blind-tiles", nargs="+", default=("11-SW-3B", "11-SW-4D"))
    parser.add_argument("--blind-count", type=int, default=40)
    args = parser.parse_args()
    paths = sorted(path for path in args.source_root.glob("*/BUILDING/*/*.gltf")
                   if not path.name.startswith("._") and path.stem == path.parent.name)
    catalog = []
    for index, path in enumerate(paths, 1):
        catalog.append(inspect_asset(path, args.source_root))
        if index % 250 == 0:
            print(f"catalog {index}/{len(paths)}", flush=True)
    qualified = [record for record in catalog if record["status"] != "unusable"]
    blind_tiles = set(args.blind_tiles)
    main_pool = [record for record in qualified if record["tile"] not in blind_tiles]
    blind_pool = [record for record in qualified if record["tile"] in blind_tiles]
    if len(main_pool) < sum(MAIN_SPLITS.values()) or len(blind_pool) < args.blind_count:
        raise RuntimeError(f"Insufficient assets: main={len(main_pool)}, blind={len(blind_pool)}")
    existing = json.loads(args.existing_manifest.read_text(encoding="utf-8"))["records"]
    existing_ids = {record["asset_id"] for record in existing}
    # Preserve qualified legacy assets from main tiles, then fill by diversity.
    legacy = [record for record in main_pool if record["asset_id"] in existing_ids]
    selected_ids = {record["asset_id"] for record in legacy}
    fill = [record for record in diversity_order(main_pool, args.seed)
            if record["asset_id"] not in selected_ids]
    main_selected = (diversity_order(legacy, args.seed) + fill)[:sum(MAIN_SPLITS.values())]
    main = assign_main_splits(main_selected, args.seed)
    blind = [{**record, "split": "blind"} for record in
             diversity_order(blind_pool, args.seed + 1)[:args.blind_count]]
    selected = main + sorted(blind, key=lambda record: (record["tile"], record["asset_id"]))
    keys = [(record["tile"], record["asset_id"]) for record in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate selected asset")
    counts = {split: sum(record["split"] == split for record in selected)
              for split in (*MAIN_SPLITS, "blind")}
    payload = {
        "schema_version": 1, "seed": args.seed, "status": "CATALOG_SELECTION_PENDING_FULL_PARSE",
        "protocol": "300 main buildings with mixed tiles and disjoint IDs; extra Blind from wholly unseen tiles",
        "source_root": str(args.source_root), "blind_tiles": sorted(blind_tiles),
        "catalog_counts": {
            "total": len(catalog), "qualified": len(qualified),
            "usable_natural_defect": sum(r["status"] == "usable_natural_defect" for r in catalog),
            "unusable": sum(r["status"] == "unusable" for r in catalog),
        },
        "selection_counts": counts,
        "legacy_selected": sum(record["asset_id"] in existing_ids for record in selected),
        "records": selected,
        "unusable_records": [record for record in catalog if record["status"] == "unusable"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"catalog": payload["catalog_counts"], "selection": counts,
                      "legacy_selected": payload["legacy_selected"],
                      "blind_tiles": payload["blind_tiles"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
