#!/usr/bin/env python3
"""Stage manifest-selected official glTF packages into the independent data tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.integrity import asset_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-project-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    seen, output, errors = set(), [], []
    for manifest in args.manifests:
        for record in json.loads(manifest.read_text(encoding="utf-8"))["records"]:
            if record["asset_id"] in seen:
                continue
            seen.add(record["asset_id"])
            source = args.legacy_project_root / record["gltf_path"]
            destination = (args.destination_root / record["sheet"] / record.get("class_name", "BUILDING")
                           / record["asset_id"])
            try:
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source.parent, destination)
                target = destination / f"{record['asset_id']}.gltf"
                digest, dependencies = asset_digest(target)
                mesh = GltfReader(target).load_mesh(include_texture=True)
                output.append({"asset_id": record["asset_id"], "sheet": record["sheet"],
                               "split": record["split"], "gltf_path": str(target),
                               "asset_digest": digest, "dependencies": dependencies,
                               "vertex_count": len(mesh.vertices), "face_count": len(mesh.faces)})
                print(f"ok {len(output)}/{len(seen)} {record['asset_id']}", flush=True)
            except Exception as error:
                errors.append({"asset_id": record["asset_id"], "error": repr(error)})
    report = {"schema_version": 1, "passed": not errors, "expected": len(seen),
              "copied": len(output), "errors": errors, "records": output}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "expected": len(seen), "copied": len(output), "errors": errors}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
