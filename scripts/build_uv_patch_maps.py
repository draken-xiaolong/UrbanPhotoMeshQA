#!/usr/bin/env python3
"""Build deterministic Mesh-Patch to triangle/UV/material sidecars.

The sidecars contain no quality labels.  Clean/attacked pairs are only needed
later when full-reference local texture targets are constructed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.integrity import sha256_file
from urbanphotomeshqa.patches import patch_layout, topological_patch_layout


def source_path(data_root: Path, row: dict) -> Path:
    return (data_root / "HK3D-Individualised" / row["sheet"] / row.get("class_name", "BUILDING")
            / row["asset_id"] / f"{row['asset_id']}.gltf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--asset-id", action="append", default=[],
                        help="Limit to one or more assets for a Pilot")
    parser.add_argument("--patches", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--layout-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    requested = set(args.asset_id)
    rows = [row for row in payload["records"] if not requested or row["asset_id"] in requested]
    if requested and requested != {row["asset_id"] for row in rows}:
        missing = sorted(requested - {row["asset_id"] for row in rows})
        raise ValueError(f"Unknown asset ids: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = []
    for row in rows:
        gltf = source_path(args.data_root, row)
        mesh = GltfReader(gltf).load_mesh(include_texture=True)
        if args.layout_version == "v1":
            descriptors, patch_mask, face_indices, face_mask = patch_layout(
                mesh, args.patches, args.neighbors)
            safe = np.maximum(face_indices, 0)
            uv_triangles = np.asarray(mesh.texcoords, np.float32)[mesh.faces[safe]]
            uv_valid = face_mask & np.isfinite(uv_triangles).all(axis=(2, 3))
            uv_triangles[~uv_valid] = np.nan
            materials = np.asarray(mesh.face_materials, np.int32)[safe]
            materials[~face_mask] = -1
            arrays = {"descriptors": descriptors, "patch_mask": patch_mask,
                      "face_indices": face_indices, "face_mask": face_mask,
                      "uv_triangles": uv_triangles, "uv_valid": uv_valid,
                      "face_materials": materials}
            valid_faces, total_faces = int(uv_valid.sum()), int(face_mask.sum())
            coverage = len(np.unique(face_indices[face_mask])) / max(len(mesh.faces), 1)
        else:
            layout = topological_patch_layout(mesh, args.patches)
            texcoords = (np.full((len(mesh.vertices), 2), np.nan, np.float32)
                         if mesh.texcoords is None else np.asarray(mesh.texcoords, np.float32))
            uv_triangles = texcoords[np.asarray(mesh.faces, np.int64)]
            uv_valid = np.isfinite(uv_triangles).all(axis=(1, 2))
            uv_triangles[~uv_valid] = np.nan
            arrays = {**layout, "uv_triangles": uv_triangles, "uv_valid": uv_valid,
                      "face_materials": np.asarray(mesh.face_materials, np.int32)}
            valid_faces, total_faces = int(uv_valid.sum()), len(mesh.faces)
            coverage = len(np.unique(layout["patch_face_indices"])) / max(len(mesh.faces), 1)
        output = args.output_dir / f"{row['asset_id']}.npz"
        np.savez_compressed(
            output,
            asset_id=np.asarray(row["asset_id"]),
            sheet=np.asarray(row["sheet"]),
            source_gltf_sha256=np.asarray(sha256_file(gltf)),
            layout_version=np.asarray(args.layout_version),
            **arrays,
        )
        audit.append({
            "asset_id": row["asset_id"], "sheet": row["sheet"],
            "source_gltf": str(gltf), "sidecar": str(output),
            "patches": int(arrays["patch_mask"].sum()), "patch_faces": total_faces,
            "uv_valid_patch_faces": valid_faces,
            "uv_valid_rate": valid_faces / max(total_faces, 1),
            "face_coverage": coverage,
            "active_patches": int(arrays["patch_mask"].sum()),
            "connected_components": (int(arrays["connected_components"])
                                     if args.layout_version == "v2" else None),
            "virtual_bridge_count": (int(arrays["virtual_bridge_count"])
                                     if args.layout_version == "v2" else None),
            "patch_area_cv": (float(arrays["patch_area"][arrays["patch_mask"]].std()
                                    / max(arrays["patch_area"][arrays["patch_mask"]].mean(), 1e-12))
                              if args.layout_version == "v2" else None),
        })
        if not args.quiet:
            print(f"ok {len(audit)}/{len(rows)} {row['asset_id']} uv={valid_faces}/{total_faces}", flush=True)

    passed = all(row["uv_valid_rate"] > 0.0 for row in audit)
    if args.layout_version == "v2":
        passed = passed and all(abs(row["face_coverage"] - 1.0) < 1e-12 for row in audit)
    report = {
        "schema_version": 1,
        "status": "PASSED" if passed else "FAILED",
        "scope": "pilot" if requested else "full-clean-source",
        "patches": args.patches, "neighbors": args.neighbors,
        "layout_version": args.layout_version,
        "records": audit,
        "mean_uv_valid_rate": float(np.mean([row["uv_valid_rate"] for row in audit])),
        "mean_face_coverage": float(np.mean([row["face_coverage"] for row in audit])),
        "max_active_patches": max(row["active_patches"] for row in audit),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
