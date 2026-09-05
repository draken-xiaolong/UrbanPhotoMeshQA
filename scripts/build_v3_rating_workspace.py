#!/usr/bin/env python3
"""Create a flat, self-contained viewing/rating workspace from a V3 manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.integrity import asset_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resources = args.output_dir / "_resources"
    resources.mkdir(exist_ok=True)
    index = []

    for record in payload["records"]:
        source = Path(record["gltf_path"])
        variant = record["variant"]
        grade = int(record["human_rating_initial"])
        root = json.loads(source.read_text(encoding="utf-8"))
        variant_resources = resources / variant
        if variant_resources.exists():
            shutil.rmtree(variant_resources, ignore_errors=True)
        variant_resources.mkdir(parents=True, exist_ok=True)

        for item_type in ("buffers", "images"):
            for item_index, item in enumerate(root.get(item_type, [])):
                uri = item.get("uri")
                if not uri or uri.startswith("data:"):
                    raise ValueError(f"Unsupported URI in {source}: {uri}")
                original = (source.parent / uri).resolve()
                category = "buffers" if item_type == "buffers" else "textures"
                target_dir = variant_resources / category
                target_dir.mkdir(exist_ok=True)
                suffix = original.suffix.lower() or (".bin" if item_type == "buffers" else ".png")
                target = target_dir / f"{item_index:02d}{suffix}"
                shutil.copy2(original, target)
                item["uri"] = str(target.relative_to(args.output_dir))

        filename = f"scale{grade}_{variant}.gltf"
        output = args.output_dir / filename
        output.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
        digest, dependencies = asset_digest(output)
        mesh = GltfReader(output).load_mesh(include_texture=True)
        index.append({
            "asset_id": record["asset_id"],
            "variant": variant,
            "initial_scale": grade,
            "rating_status": record.get("rating_status", "legacy_initial_suggestion"),
            "rating_filename": filename,
            "canonical_gltf_path": str(source),
            "rating_asset_digest": digest,
            "dependency_count": len(dependencies),
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
        })

    index_path = args.output_dir / "rating_workspace_index.json"
    index_path.write_text(json.dumps({
        "schema_version": 1,
        "instruction": "Change only the scaleN_ filename prefix to revise the human rating.",
        "records": index,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASSED", "models": len(index), "index": str(index_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
