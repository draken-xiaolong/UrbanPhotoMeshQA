"""Minimal external-buffer glTF writer for textured triangle meshes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gltf import MeshAsset


def _append_aligned(buffer: bytearray, array: np.ndarray) -> tuple[int, int]:
    while len(buffer) % 4:
        buffer.append(0)
    offset = len(buffer)
    payload = np.ascontiguousarray(array).tobytes()
    buffer.extend(payload)
    return offset, len(payload)


def export_textured_gltf(asset: MeshAsset, output: Path, texture_names: list[str]) -> None:
    if asset.texcoords is None or len(asset.texcoords) != len(asset.vertices):
        raise ValueError("Textured glTF export requires vertex-aligned UV coordinates")
    output.parent.mkdir(parents=True, exist_ok=True)
    binary = bytearray()
    views: list[dict[str, object]] = []
    accessors: list[dict[str, object]] = []

    def add(array, component_type, accessor_type, bounds=False):
        offset, length = _append_aligned(binary, array)
        view = len(views)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": length})
        accessor: dict[str, object] = {
            "bufferView": view,
            "componentType": component_type,
            "count": int(len(array)),
            "type": accessor_type,
        }
        if bounds:
            accessor["min"] = np.min(array, axis=0).astype(float).tolist()
            accessor["max"] = np.max(array, axis=0).astype(float).tolist()
        accessors.append(accessor)
        return len(accessors) - 1

    primitives = []
    for material in sorted(np.unique(asset.face_materials).tolist()):
        if material < 0:
            raise ValueError("Every face must retain a valid material index")
        material_faces = np.asarray(asset.faces[asset.face_materials == material], dtype=np.int64)
        if not len(material_faces):
            continue
        used, remapped = np.unique(material_faces.reshape(-1), return_inverse=True)
        positions = np.asarray(asset.vertices[used], dtype="<f4")
        normals = np.asarray(asset.normals[used], dtype="<f4")
        texcoords = np.asarray(asset.texcoords[used], dtype="<f4")
        indices = np.asarray(remapped, dtype="<u4")
        primitives.append({
            "attributes": {
                "POSITION": add(positions, 5126, "VEC3", True),
                "NORMAL": add(normals, 5126, "VEC3"),
                "TEXCOORD_0": add(texcoords, 5126, "VEC2"),
            },
            "indices": add(indices, 5125, "SCALAR"),
            "material": int(material),
            "mode": 4,
        })

    material_count = max(int(np.max(asset.face_materials)) + 1, len(texture_names))
    images = [{"uri": f"textures/{name}"} for name in texture_names]
    textures = [{"source": index} for index in range(len(images))]
    materials = []
    for index in range(material_count):
        pbr: dict[str, object] = {"metallicFactor": 0.0, "roughnessFactor": 1.0}
        if index < len(textures):
            pbr["baseColorTexture"] = {"index": index}
        materials.append({"name": f"material_{index:02d}", "doubleSided": True, "pbrMetallicRoughness": pbr})

    bin_name = f"{output.stem}.bin"
    root = {
        "asset": {"version": "2.0", "generator": "UrbanPhotoMeshQA real glTF attack exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"name": output.stem, "primitives": primitives}],
        "buffers": [{"uri": bin_name, "byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
        "images": images,
        "textures": textures,
        "materials": materials,
    }
    output.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".bin").write_bytes(binary)
