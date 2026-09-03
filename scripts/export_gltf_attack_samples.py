#!/usr/bin/env python3
"""Export UV-preserving Mesh attacks as inspectable glTF + BIN + shared textures."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from urbanphotomeshqa.gltf import GltfReader, MeshAsset  # noqa: E402
from urbanphotomeshqa.mesh_attacks import topology_stats  # noqa: E402
from urbanphotomeshqa.texture import (  # noqa: E402
    STANDARD_DIRECTIONS, apply_uv_preserving_attack, render_textured_view,
)


ATTACKS = {
    "connected_crop": (0.10, 0.30, 0.50),
    "hole": (0.05, 0.15, 0.30),
    "retriangulate": (0.10, 0.25, 0.45),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gltf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def aligned_append(buffer: bytearray, array: np.ndarray) -> tuple[int, int]:
    while len(buffer) % 4:
        buffer.append(0)
    offset = len(buffer)
    payload = np.ascontiguousarray(array).tobytes()
    buffer.extend(payload)
    return offset, len(payload)


def export_gltf(asset: MeshAsset, output: Path, texture_names: list[str]):
    buffer = bytearray(); views = []; accessors = []

    def add(array, component_type, accessor_type, include_bounds=False):
        offset, length = aligned_append(buffer, array)
        view_index = len(views)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": length})
        accessor = {"bufferView": view_index, "componentType": component_type,
                    "count": int(len(array)), "type": accessor_type}
        if include_bounds:
            accessor["min"] = np.min(array, axis=0).astype(float).tolist()
            accessor["max"] = np.max(array, axis=0).astype(float).tolist()
        accessors.append(accessor)
        return len(accessors) - 1

    primitives = []
    for material in sorted(np.unique(asset.face_materials).tolist()):
        material_faces = np.asarray(asset.faces[asset.face_materials == material], dtype=np.int64)
        used, remapped = np.unique(material_faces.reshape(-1), return_inverse=True)
        positions = np.asarray(asset.vertices[used], dtype="<f4")
        normals = np.asarray(asset.normals[used], dtype="<f4")
        texcoords = np.asarray(asset.texcoords[used], dtype="<f4")
        indices = np.asarray(remapped, dtype="<u4")
        position_accessor = add(positions, 5126, "VEC3", True)
        normal_accessor = add(normals, 5126, "VEC3")
        uv_accessor = add(texcoords, 5126, "VEC2")
        index_accessor = add(indices, 5125, "SCALAR")
        primitives.append({
            "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor,
                           "TEXCOORD_0": uv_accessor},
            "indices": index_accessor, "material": int(material), "mode": 4,
        })
    images = [{"uri": f"textures/{name}"} for name in texture_names]
    textures = [{"source": index} for index in range(len(images))]
    materials = []
    count = int(max(np.max(asset.face_materials), len(texture_names) - 1) + 1)
    for index in range(count):
        material = {"name": f"material_{index:02d}", "doubleSided": True,
                    "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}}
        if index < len(textures):
            material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": index}
            if texture_names[index].lower().endswith(".png"):
                material["alphaMode"] = "MASK"; material["alphaCutoff"] = 0.5
        materials.append(material)
    binary_name = output.with_suffix(".bin").name
    document = {
        "asset": {"version": "2.0", "generator": "UrbanPhotoMeshQA attack sample exporter"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": output.stem}],
        "meshes": [{"name": output.stem, "primitives": primitives}],
        "buffers": [{"uri": binary_name, "byteLength": len(buffer)}],
        "bufferViews": views, "accessors": accessors,
        "images": images, "textures": textures, "materials": materials,
        "extras": {"attack": asset.metadata.get("attack", "clean"),
                   "severity": float(asset.metadata.get("severity", 0.0))},
    }
    output.with_suffix(".bin").write_bytes(buffer)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    texture_dir = args.output_dir / "textures"; texture_dir.mkdir(exist_ok=True)
    source = GltfReader(args.gltf).load_mesh(include_texture=True)
    texture_paths = [Path(value) for value in source.metadata["material_texture_paths"]]
    texture_names = [path.name for path in texture_paths]
    for path in texture_paths:
        shutil.copy2(path, texture_dir / path.name)
    variants = [("clean", source)]
    for attack, severities in ATTACKS.items():
        for level_index, severity in enumerate(severities):
            level = ("light", "medium", "heavy")[level_index]
            variants.append((f"{attack}_{level}_{severity:.2f}",
                             apply_uv_preserving_attack(source, attack, severity,
                                                        args.seed + len(variants) * 97)))
    records = []
    label_height = 25
    contact = Image.new("RGB", (len(STANDARD_DIRECTIONS) * args.render_size,
                                 len(variants) * (args.render_size + label_height)), "white")
    draw = ImageDraw.Draw(contact)
    preview_dir = args.output_dir / "previews"; preview_dir.mkdir(exist_ok=True)
    for row, (name, asset) in enumerate(variants):
        output = args.output_dir / f"{name}.gltf"
        export_gltf(asset, output, texture_names)
        preview_paths = []
        for column, direction in enumerate(STANDARD_DIRECTIONS):
            rendered = render_textured_view(asset, direction=direction, size=args.render_size)
            preview = preview_dir / f"{name}_view{column}.png"
            Image.fromarray(rendered).save(preview); preview_paths.append(str(preview.relative_to(args.output_dir)))
            contact.paste(Image.fromarray(rendered),
                          (column * args.render_size, row * (args.render_size + label_height) + label_height))
        draw.text((5, row * (args.render_size + label_height) + 5), name, fill="black")
        records.append({"name": name, "gltf": output.name, "bin": output.with_suffix('.bin').name,
                        "vertices": int(len(asset.vertices)), "faces": int(len(asset.faces)),
                        "attack": asset.metadata.get("attack", "clean"),
                        "severity": float(asset.metadata.get("severity", 0.0)),
                        "topology": topology_stats(asset), "previews": preview_paths})
    contact.save(args.output_dir / "contact_sheet.png")
    manifest = {"source": str(args.gltf), "asset_id": source.metadata.get("asset_id"),
                "seed": args.seed, "textures_shared": texture_names,
                "severity_meaning": {"connected_crop": "fraction removed; one connected region retained",
                                     "hole": "connected face fraction removed",
                                     "retriangulate": "face fraction split into three coplanar faces"},
                "records": records}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "asset_id": manifest["asset_id"],
                      "variants": len(records), "contact_sheet": "contact_sheet.png"}, indent=2))


if __name__ == "__main__":
    main()
