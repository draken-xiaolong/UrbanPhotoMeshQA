#!/usr/bin/env python3
"""Audit clean meshes before choosing a topology-preserving patch capacity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.patches import _components, _geometric_face_adjacency


def source_path(data_root: Path, row: dict) -> Path:
    return (data_root / "HK3D-Individualised" / row["sheet"]
            / row.get("class_name", "BUILDING") / row["asset_id"]
            / f"{row['asset_id']}.gltf")


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    records = []
    for row in payload["records"]:
        gltf = source_path(args.data_root, row)
        mesh = GltfReader(gltf).load_mesh(include_texture=False)
        components = _components(_geometric_face_adjacency(mesh))
        sizes = sorted((len(value) for value in components), reverse=True)
        records.append({
            "asset_id": row["asset_id"],
            "sheet": row["sheet"],
            "faces": int(len(mesh.faces)),
            "edge_connected_components": len(components),
            "largest_component_faces": sizes[0],
            "singleton_components": int(sum(value == 1 for value in sizes)),
        })

    counts = [row["edge_connected_components"] for row in records]
    faces = [row["faces"] for row in records]
    summary = {
        "schema_version": 1,
        "assets": len(records),
        "component_count": {
            "min": min(counts), "median": percentile(counts, 50),
            "p90": percentile(counts, 90), "p95": percentile(counts, 95),
            "max": max(counts),
            "over_16": sum(value > 16 for value in counts),
            "over_32": sum(value > 32 for value in counts),
            "over_64": sum(value > 64 for value in counts),
            "over_128": sum(value > 128 for value in counts),
        },
        "face_count": {
            "min": min(faces), "median": percentile(faces, 50),
            "p95": percentile(faces, 95), "max": max(faces),
        },
        "largest_component_assets": sorted(
            records, key=lambda row: (-row["edge_connected_components"], row["asset_id"])
        )[:20],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"records", "largest_component_assets"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
