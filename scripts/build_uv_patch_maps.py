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
from urbanphotomeshqa.patches import patch_layout


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
    args = parser.parse_args()

    payload = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    requested = set(args.asset_id)
    rows = [row for row in payload["records"] if not requested or row["asset_id"] in requested]
    if requested != {row["asset_id"] for row in rows}:
        missing = sorted(requested - {row["asset_id"] for row in rows})
        raise ValueError(f"Unknown asset ids: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = []
    for row in rows:
        gltf = source_path(args.data_root, row)
        mesh = GltfReader(gltf).load_mesh(include_texture=True)
        descriptors, patch_mask, face_indices, face_mask = patch_layout(
            mesh, args.patches, args.neighbors)
        safe = np.maximum(face_indices, 0)
        uv_triangles = np.asarray(mesh.texcoords, np.float32)[mesh.faces[safe]]
        uv_valid = face_mask & np.isfinite(uv_triangles).all(axis=(2, 3))
        uv_triangles[~uv_valid] = np.nan
        materials = np.asarray(mesh.face_materials, np.int32)[safe]
        materials[~face_mask] = -1
        output = args.output_dir / f"{row['asset_id']}.npz"
        np.savez_compressed(
            output,
            asset_id=np.asarray(row["asset_id"]),
            sheet=np.asarray(row["sheet"]),
            source_gltf_sha256=np.asarray(sha256_file(gltf)),
            descriptors=descriptors,
            patch_mask=patch_mask,
            face_indices=face_indices,
            face_mask=face_mask,
            uv_triangles=uv_triangles,
            uv_valid=uv_valid,
            face_materials=materials,
        )
        valid_faces = int(uv_valid.sum())
        total_faces = int(face_mask.sum())
        audit.append({
            "asset_id": row["asset_id"], "sheet": row["sheet"],
            "source_gltf": str(gltf), "sidecar": str(output),
            "patches": int(patch_mask.sum()), "patch_faces": total_faces,
            "uv_valid_patch_faces": valid_faces,
            "uv_valid_rate": valid_faces / max(total_faces, 1),
        })
        print(f"ok {len(audit)}/{len(rows)} {row['asset_id']} uv={valid_faces}/{total_faces}", flush=True)

    report = {
        "schema_version": 1,
        "status": "PASSED" if all(row["uv_valid_rate"] > 0.0 for row in audit) else "FAILED",
        "scope": "pilot" if requested else "full-clean-source",
        "patches": args.patches, "neighbors": args.neighbors,
        "records": audit,
        "mean_uv_valid_rate": float(np.mean([row["uv_valid_rate"] for row in audit])),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
