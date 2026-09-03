#!/usr/bin/env python3
"""Fully parse and digest copied Iteration 2 source assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbanphotomeshqa.gltf import GltfReader  # noqa: E402
from urbanphotomeshqa.integrity import asset_digest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path,
        help="Optional original source root. If omitted, verify copied assets against "
             "asset_digest values already stored in --selection.",
    )
    parser.add_argument("--copied-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    records = []; errors = []; digests = {}
    for index, record in enumerate(selection["records"], 1):
        relative = Path(record["source_gltf"])
        copied = args.copied_root / relative
        try:
            source_dependencies = None
            if args.source_root is not None:
                source = args.source_root / relative
                expected_digest, source_dependencies = asset_digest(source)
            else:
                expected_digest = record.get("asset_digest")
                if not expected_digest:
                    raise ValueError(
                        "selection record has no asset_digest and --source-root was omitted"
                    )
            copied_digest, copied_dependencies = asset_digest(copied)
            if expected_digest != copied_digest:
                raise ValueError("expected/copied asset digest mismatch")
            mesh = GltfReader(copied).load_mesh(include_texture=True)
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                raise ValueError("empty parsed mesh")
            if mesh.texcoords is None or len(mesh.texcoords) != len(mesh.vertices):
                raise ValueError("missing vertex-aligned TEXCOORD_0")
            digest_duplicate = copied_digest in digests
            if digest_duplicate:
                errors.append(f"duplicate asset digest: {record['asset_id']} == {digests[copied_digest]}")
            else:
                digests[copied_digest] = record["asset_id"]
            vertices = np.asarray(mesh.vertices, np.float64)
            output = {
                **record,
                "gltf_path": str(Path(args.copied_root.name) / relative),
                "asset_digest": copied_digest,
                "dependency_count_full": len(copied_dependencies),
                "parsed_vertex_count": int(len(mesh.vertices)),
                "parsed_face_count": int(len(mesh.faces)),
                "material_count": int(len(set(np.asarray(mesh.face_materials).tolist()))),
                "bounding_diagonal": float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))),
                "digest_duplicate": digest_duplicate,
            }
            if source_dependencies is not None and len(source_dependencies) != len(copied_dependencies):
                raise ValueError("source/copy dependency count mismatch")
            records.append(output)
        except Exception as error:
            errors.append(f"{record['split']}/{record['tile']}/{record['asset_id']}: {type(error).__name__}: {error}")
        if index % 25 == 0 or index == len(selection["records"]):
            print(f"audit {index}/{len(selection['records'])} errors={len(errors)}", flush=True)
    counts = {split: sum(record["split"] == split for record in records)
              for split in ("train", "val", "test", "blind")}
    payload = {
        "schema_version": 1, "status": "PASSED" if not errors and len(records) == 340 else "FAILED",
        "selection_manifest": str(args.selection),
        "source_root": str(args.source_root) if args.source_root is not None else None,
        "copied_root": str(args.copied_root), "counts": counts,
        "records": records, "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "counts": counts, "errors": errors[:5]}, ensure_ascii=False))
    if payload["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
