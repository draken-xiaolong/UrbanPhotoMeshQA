#!/usr/bin/env python3
"""Audit the 184-asset clean glTF subset and emit a content-addressed manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from urbanphotomeshqa.gltf import GltfReader  # noqa: E402
from urbanphotomeshqa.integrity import asset_digest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    failures = []
    for index, row in enumerate(source["records"]):
        relative = Path(row["sheet"]) / row["class_name"] / row["asset_id"] / f"{row['asset_id']}.gltf"
        gltf = args.data_root / relative
        try:
            digest, files = asset_digest(gltf)
            mesh = GltfReader(gltf).load_mesh(include_texture=True)
            records.append({
                **row,
                "relative_gltf": relative.as_posix(),
                "asset_digest": digest,
                "files": files,
                "parsed_vertices": int(len(mesh.vertices)),
                "parsed_faces": int(len(mesh.faces)),
                "finite_uv_fraction": float(0.0 if mesh.texcoords is None else __import__("numpy").isfinite(mesh.texcoords).all(1).mean()),
            })
        except Exception as error:
            failures.append({"asset_id": row.get("asset_id"), "error": f"{type(error).__name__}: {error}"})
        if index == 0 or (index + 1) % 20 == 0:
            print(f"audited {index + 1}/{len(source['records'])}", flush=True)
    output = {
        "schema_version": 1,
        "data_root": str(args.data_root),
        "source_manifest": str(args.manifest),
        "records": records,
        "failures": failures,
        "summary": {"expected": len(source["records"]), "passed": len(records), "failed": len(failures)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"]))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
