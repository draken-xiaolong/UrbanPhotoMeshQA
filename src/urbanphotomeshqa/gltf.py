from __future__ import annotations

import json
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_COMPONENT_DTYPES = {
    5120: np.dtype("<i1"),
    5121: np.dtype("<u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


@dataclass
class MeshAsset:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    face_materials: np.ndarray
    metadata: dict[str, Any]
    texcoords: np.ndarray | None = None


def _node_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        # glTF matrices are column-major.
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4, order="F")
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    x, y, z, w = np.asarray(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


class GltfReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.root = json.loads(self.path.read_text(encoding="utf-8"))
        self.buffers = []
        for buffer in self.root.get("buffers", []):
            uri = buffer.get("uri")
            if not uri or uri.startswith("data:"):
                raise ValueError(f"Only external binary buffers are supported: {self.path}")
            self.buffers.append((self.path.parent / uri).read_bytes())

    def accessor(self, index: int) -> np.ndarray:
        accessor = self.root["accessors"][index]
        view = self.root["bufferViews"][accessor["bufferView"]]
        dtype = _COMPONENT_DTYPES[accessor["componentType"]]
        components = _TYPE_COMPONENTS[accessor["type"]]
        count = int(accessor["count"])
        offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        stride = int(view.get("byteStride", dtype.itemsize * components))
        raw = self.buffers[int(view.get("buffer", 0))]
        if stride == dtype.itemsize * components:
            values = np.frombuffer(raw, dtype=dtype, count=count * components, offset=offset)
            return values.reshape(count, components).copy()
        values = np.ndarray(
            shape=(count, components),
            dtype=dtype,
            buffer=raw,
            offset=offset,
            strides=(stride, dtype.itemsize),
        )
        return values.copy()

    def load_mesh(self, include_texture: bool = False) -> MeshAsset:
        vertices_parts: list[np.ndarray] = []
        normals_parts: list[np.ndarray] = []
        faces_parts: list[np.ndarray] = []
        material_parts: list[np.ndarray] = []
        texcoord_parts: list[np.ndarray] = []

        nodes = self.root.get("nodes", [])
        scene_index = int(self.root.get("scene", 0))
        scenes = self.root.get("scenes", [{"nodes": list(range(len(nodes)))}])

        def visit(node_index: int, parent: np.ndarray) -> None:
            node = nodes[node_index]
            transform = parent @ _node_matrix(node)
            if "mesh" in node:
                mesh = self.root["meshes"][node["mesh"]]
                for primitive in mesh.get("primitives", []):
                    if int(primitive.get("mode", 4)) != 4:
                        continue
                    attrs = primitive["attributes"]
                    positions = self.accessor(attrs["POSITION"]).astype(np.float64)
                    homogeneous = np.column_stack([positions, np.ones(len(positions))])
                    positions = (homogeneous @ transform.T)[:, :3]

                    if "NORMAL" in attrs:
                        normals = self.accessor(attrs["NORMAL"]).astype(np.float64)
                        normal_matrix = np.linalg.inv(transform[:3, :3]).T
                        normals = normals @ normal_matrix.T
                        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                        normals = normals / np.maximum(lengths, 1e-12)
                    else:
                        normals = np.zeros_like(positions)

                    if include_texture and "TEXCOORD_0" in attrs:
                        texcoords = self.accessor(attrs["TEXCOORD_0"]).astype(np.float64)[:, :2]
                    elif include_texture:
                        texcoords = np.full((len(positions), 2), np.nan, dtype=np.float64)

                    if "indices" in primitive:
                        indices = self.accessor(primitive["indices"]).reshape(-1).astype(np.int64)
                    else:
                        indices = np.arange(len(positions), dtype=np.int64)
                    if len(indices) % 3:
                        raise ValueError(f"Triangle index count is not divisible by 3: {self.path}")
                    offset = sum(len(part) for part in vertices_parts)
                    faces = indices.reshape(-1, 3) + offset
                    vertices_parts.append(positions)
                    normals_parts.append(normals)
                    faces_parts.append(faces)
                    material_parts.append(
                        np.full(len(faces), int(primitive.get("material", -1)), dtype=np.int32)
                    )
                    if include_texture:
                        texcoord_parts.append(texcoords)
            for child in node.get("children", []):
                visit(int(child), transform)

        for root_node in scenes[scene_index].get("nodes", []):
            visit(int(root_node), np.eye(4, dtype=np.float64))

        if not vertices_parts or not faces_parts:
            raise ValueError(f"No triangle primitives found: {self.path}")
        vertices = np.concatenate(vertices_parts, axis=0)
        faces = np.concatenate(faces_parts, axis=0)
        normals = np.concatenate(normals_parts, axis=0)
        face_materials = np.concatenate(material_parts, axis=0)
        material_texture_paths: list[str | None] = []
        material_profiles: list[dict[str, Any]] = []
        textures = self.root.get("textures", [])
        images = self.root.get("images", [])
        for material in self.root.get("materials", []):
            texture_info = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
            path = None
            sampler = None
            if texture_info is not None:
                texture_index = int(texture_info["index"])
                if 0 <= texture_index < len(textures):
                    source_index = int(textures[texture_index].get("source", -1))
                    sampler_index = int(textures[texture_index].get("sampler", -1))
                    if 0 <= source_index < len(images) and images[source_index].get("uri"):
                        path = str((self.path.parent / images[source_index]["uri"]).resolve())
                    samplers = self.root.get("samplers", [])
                    if 0 <= sampler_index < len(samplers):
                        sampler = copy.deepcopy(samplers[sampler_index])
            material_texture_paths.append(path)
            material_profiles.append({
                "name": material.get("name"),
                "doubleSided": bool(material.get("doubleSided", False)),
                "alphaMode": material.get("alphaMode", "OPAQUE"),
                "alphaCutoff": material.get("alphaCutoff"),
                "emissiveFactor": copy.deepcopy(material.get("emissiveFactor")),
                "pbrMetallicRoughness": copy.deepcopy(material.get("pbrMetallicRoughness", {})),
                "baseColorSampler": sampler,
            })
        return MeshAsset(
            vertices=vertices,
            faces=faces,
            normals=normals,
            face_materials=face_materials,
            metadata={
                "asset_id": self.path.stem,
                "gltf_path": str(self.path),
                "vertex_count": int(len(vertices)),
                "face_count": int(len(faces)),
                "material_count": int(len(self.root.get("materials", []))),
                "image_count": int(len(self.root.get("images", []))),
                "material_texture_paths": material_texture_paths if include_texture else [],
                "material_profiles": material_profiles if include_texture else [],
            },
            texcoords=np.concatenate(texcoord_parts, axis=0) if include_texture else None,
        )


def sample_surface(asset: MeshAsset, count: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    triangles = asset.vertices[asset.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    if not np.any(valid):
        raise ValueError(f"Mesh has no non-degenerate faces: {asset.metadata['gltf_path']}")
    probabilities = np.where(valid, double_area, 0.0)
    probabilities /= probabilities.sum()
    selected = rng.choice(len(triangles), size=count, replace=True, p=probabilities)
    tri = triangles[selected]
    r1 = np.sqrt(rng.random((count, 1)))
    r2 = rng.random((count, 1))
    bary = np.concatenate([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1)
    points = np.sum(tri * bary[:, :, None], axis=1)

    vertex_normals = asset.normals[asset.faces[selected]]
    sampled_normals = np.sum(vertex_normals * bary[:, :, None], axis=1)
    missing = np.linalg.norm(sampled_normals, axis=1) < 1e-8
    face_normals = cross[selected] / np.maximum(double_area[selected, None], 1e-12)
    sampled_normals[missing] = face_normals[missing]
    sampled_normals /= np.maximum(np.linalg.norm(sampled_normals, axis=1, keepdims=True), 1e-12)

    minimum = asset.vertices.min(axis=0)
    maximum = asset.vertices.max(axis=0)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    scale = float(max(np.linalg.norm(extent), 1e-8))
    normalized = (points - center) / scale
    samples = np.concatenate([normalized, sampled_normals], axis=1).astype(np.float32)
    stats = {
        **asset.metadata,
        "bbox_min": minimum.tolist(),
        "bbox_max": maximum.tolist(),
        "bbox_extent": extent.tolist(),
        "normalization_center": center.tolist(),
        "normalization_scale": scale,
        "surface_area": float(double_area.sum() * 0.5),
        "degenerate_face_count": int((~valid).sum()),
    }
    return samples, stats
