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


def export_textured_gltf(
    asset: MeshAsset,
    output: Path,
    texture_names: list[str],
    coordinate_origin: np.ndarray | None = None,
) -> None:
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

    if coordinate_origin is None:
        coordinate_origin = 0.5 * (
            np.asarray(asset.vertices, dtype=np.float64).min(axis=0)
            + np.asarray(asset.vertices, dtype=np.float64).max(axis=0)
        )
    coordinate_origin = np.asarray(coordinate_origin, dtype=np.float64)
    if coordinate_origin.shape != (3,) or not np.isfinite(coordinate_origin).all():
        raise ValueError("coordinate_origin must be one finite XYZ vector")

    primitives = []
    for material in sorted(np.unique(asset.face_materials).tolist()):
        if material < 0:
            raise ValueError("Every face must retain a valid material index")
        material_faces = np.asarray(asset.faces[asset.face_materials == material], dtype=np.int64)
        if not len(material_faces):
            continue
        used, remapped = np.unique(material_faces.reshape(-1), return_inverse=True)
        # glTF POSITION is float32.  Keeping Hong Kong world coordinates near
        # 8e5 here would quantise geometry by centimetres.  Store local values
        # and restore the world origin through the node's JSON translation.
        positions = np.asarray(asset.vertices[used] - coordinate_origin, dtype="<f4")
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
    material_profiles = asset.metadata.get("material_profiles", [])
    samplers = []
    textures = []
    for index in range(len(images)):
        texture = {"source": index}
        profile = material_profiles[index] if index < len(material_profiles) else {}
        sampler = profile.get("baseColorSampler") if profile else None
        if sampler:
            texture["sampler"] = len(samplers)
            samplers.append(dict(sampler))
        textures.append(texture)
    materials = []
    for index in range(material_count):
        profile = material_profiles[index] if index < len(material_profiles) else {}
        pbr: dict[str, object] = dict(profile.get("pbrMetallicRoughness", {}))
        base_color_texture = dict(pbr.pop("baseColorTexture", {}))
        if not profile:
            pbr.setdefault("metallicFactor", 0.0)
            pbr.setdefault("roughnessFactor", 1.0)
        if index < len(textures):
            base_color_texture["index"] = index
            pbr["baseColorTexture"] = base_color_texture
        material = {
            "name": profile.get("name") or f"material_{index:02d}",
            "doubleSided": bool(profile.get("doubleSided", True)),
            "pbrMetallicRoughness": pbr,
        }
        if profile.get("alphaMode", "OPAQUE") != "OPAQUE":
            material["alphaMode"] = profile["alphaMode"]
        if profile.get("alphaCutoff") is not None:
            material["alphaCutoff"] = profile["alphaCutoff"]
        if profile.get("emissiveFactor") is not None:
            material["emissiveFactor"] = profile["emissiveFactor"]
        materials.append(material)

    bin_name = f"{output.stem}.bin"
    root = {
        "asset": {"version": "2.0", "generator": "UrbanPhotoMeshQA real glTF attack exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": coordinate_origin.astype(float).tolist()}],
        "meshes": [{"name": output.stem, "primitives": primitives}],
        "buffers": [{"uri": bin_name, "byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
        "images": images,
        "textures": textures,
        "materials": materials,
    }
    if samplers:
        root["samplers"] = samplers
    output.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".bin").write_bytes(binary)
