from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from typing import Any

import numpy as np

from .gltf import MeshAsset


def recompute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals for a triangle mesh."""
    normals = np.zeros_like(vertices, dtype=np.float64)
    triangles = vertices[faces]
    face_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_vectors)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def compact_mesh(asset: MeshAsset, weld_decimals: int | None = None) -> MeshAsset:
    """Remove unused vertices and optionally weld coincident glTF render vertices."""
    vertices = np.asarray(asset.vertices, dtype=np.float64)
    texcoords = None if asset.texcoords is None else np.asarray(asset.texcoords, dtype=np.float64)
    faces = np.asarray(asset.faces, dtype=np.int64)
    if weld_decimals is not None:
        _, inverse = np.unique(np.round(vertices, weld_decimals), axis=0, return_inverse=True)
        faces = inverse[faces]
        representatives = np.full(int(inverse.max()) + 1, -1, dtype=np.int64)
        representatives[inverse] = np.arange(len(inverse), dtype=np.int64)
        vertices = vertices[representatives]
        if texcoords is not None:
            # Position welding destroys UV seams. Geometry-only callers retain
            # their previous behavior, while textured protocols must compact
            # with weld_decimals=None.
            texcoords = None

    nondegenerate = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[nondegenerate]
    materials = np.asarray(asset.face_materials, dtype=np.int32)[nondegenerate]
    if not len(faces):
        raise ValueError("Mesh became empty after compaction")
    used, remapped = np.unique(faces.reshape(-1), return_inverse=True)
    vertices = vertices[used]
    if texcoords is not None:
        texcoords = texcoords[used]
    faces = remapped.reshape(-1, 3)
    return MeshAsset(
        vertices=vertices,
        faces=faces,
        normals=recompute_vertex_normals(vertices, faces),
        face_materials=materials,
        metadata=dict(asset.metadata),
        texcoords=texcoords,
    )


def face_adjacency(faces: np.ndarray) -> list[list[int]]:
    """Build edge-sharing face adjacency."""
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces[(min(int(a), int(b)), max(int(a), int(b)))].append(face_index)
    adjacency = [set() for _ in range(len(faces))]
    for attached in edge_faces.values():
        for face_index in attached:
            adjacency[face_index].update(other for other in attached if other != face_index)
    return [sorted(values) for values in adjacency]


def connected_face_patch(
    faces: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select a spatially connected patch using randomized breadth-first growth."""
    count = min(max(int(count), 1), len(faces))
    adjacency = face_adjacency(faces)
    remaining = set(range(len(faces)))
    components: list[list[int]] = []
    while remaining:
        component = [remaining.pop()]
        queue = list(component)
        while queue:
            current = queue.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        components.append(component)
    largest = max(components, key=len)
    count = min(count, len(largest))
    seed = int(rng.choice(largest))
    selected: list[int] = []
    visited = {seed}
    queue: deque[int] = deque([seed])
    while queue and len(selected) < count:
        current = queue.popleft()
        selected.append(current)
        neighbors = np.asarray(adjacency[current], dtype=np.int64)
        rng.shuffle(neighbors)
        for neighbor in neighbors.tolist():
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return np.asarray(selected, dtype=np.int64)


def _subset_faces(asset: MeshAsset, indices: np.ndarray, attack: str, severity: float) -> MeshAsset:
    result = MeshAsset(
        vertices=asset.vertices,
        faces=asset.faces[indices],
        normals=asset.normals,
        face_materials=asset.face_materials[indices],
        metadata={**asset.metadata, "attack": attack, "severity": float(severity)},
    )
    return compact_mesh(result)


def connected_crop(asset: MeshAsset, keep_fraction: float, seed: int) -> MeshAsset:
    """Keep one connected surface region, representing partial-model extraction."""
    mesh = compact_mesh(asset, weld_decimals=8)
    keep = max(4, int(round(len(mesh.faces) * keep_fraction)))
    selected = connected_face_patch(mesh.faces, keep, np.random.default_rng(seed))
    return _subset_faces(mesh, selected, "connected_crop", 1.0 - keep_fraction)


def punch_hole(asset: MeshAsset, remove_fraction: float, seed: int) -> MeshAsset:
    """Remove a connected face patch, creating missing surface/hole damage."""
    mesh = compact_mesh(asset, weld_decimals=8)
    remove = min(max(1, int(round(len(mesh.faces) * remove_fraction))), len(mesh.faces) - 4)
    selected = connected_face_patch(mesh.faces, remove, np.random.default_rng(seed))
    keep_mask = np.ones(len(mesh.faces), dtype=bool)
    keep_mask[selected] = False
    return _subset_faces(mesh, np.flatnonzero(keep_mask), "hole", remove_fraction)


def flip_face_winding(asset: MeshAsset, fraction: float, seed: int) -> MeshAsset:
    """Reverse a connected set of triangle windings, simulating normal-orientation errors."""
    mesh = compact_mesh(asset, weld_decimals=8)
    count = max(1, int(round(len(mesh.faces) * fraction)))
    selected = connected_face_patch(mesh.faces, count, np.random.default_rng(seed))
    faces = mesh.faces.copy()
    faces[selected] = faces[selected][:, [0, 2, 1]]
    return replace(
        mesh,
        faces=faces,
        normals=recompute_vertex_normals(mesh.vertices, faces),
        metadata={**mesh.metadata, "attack": "normal_flip", "severity": float(fraction)},
    )


def centroid_retriangulate(asset: MeshAsset, fraction: float, seed: int) -> MeshAsset:
    """Split selected triangles into three coplanar faces without changing the surface."""
    mesh = compact_mesh(asset, weld_decimals=8)
    count = max(1, int(round(len(mesh.faces) * fraction)))
    selected = set(connected_face_patch(mesh.faces, count, np.random.default_rng(seed)).tolist())
    vertices = mesh.vertices.tolist()
    faces: list[list[int]] = []
    materials: list[int] = []
    for index, face in enumerate(mesh.faces):
        material = int(mesh.face_materials[index])
        if index not in selected:
            faces.append(face.tolist())
            materials.append(material)
            continue
        centroid_index = len(vertices)
        vertices.append(mesh.vertices[face].mean(axis=0).tolist())
        a, b, c = face.tolist()
        faces.extend([[a, b, centroid_index], [b, c, centroid_index], [c, a, centroid_index]])
        materials.extend([material, material, material])
    vertices_array = np.asarray(vertices, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    return MeshAsset(
        vertices=vertices_array,
        faces=faces_array,
        normals=recompute_vertex_normals(vertices_array, faces_array),
        face_materials=np.asarray(materials, dtype=np.int32),
        metadata={**mesh.metadata, "attack": "retriangulate", "severity": float(fraction)},
    )


def qem_simplify(asset: MeshAsset, reduction: float) -> MeshAsset:
    """Quadric-error mesh simplification via fast-simplification/trimesh."""
    import trimesh

    mesh = compact_mesh(asset, weld_decimals=8)
    target = max(4, int(round(len(mesh.faces) * (1.0 - reduction))))
    source = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    simplified = source.simplify_quadric_decimation(face_count=target, aggression=5)
    vertices = np.asarray(simplified.vertices, dtype=np.float64)
    faces = np.asarray(simplified.faces, dtype=np.int64)
    if len(faces) < 4:
        raise ValueError("QEM produced fewer than four faces")
    return MeshAsset(
        vertices=vertices,
        faces=faces,
        normals=recompute_vertex_normals(vertices, faces),
        face_materials=np.full(len(faces), -1, dtype=np.int32),
        metadata={**mesh.metadata, "attack": "qem", "severity": float(reduction)},
    )


def topology_stats(asset: MeshAsset) -> dict[str, Any]:
    """Basic mesh/topology measurements used by the real-attack benchmark."""
    mesh = compact_mesh(asset, weld_decimals=8)
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for face in mesh.faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_counts[(min(int(a), int(b)), max(int(a), int(b)))] += 1
    adjacency = face_adjacency(mesh.faces)
    remaining = set(range(len(mesh.faces)))
    components = 0
    while remaining:
        components += 1
        queue = [remaining.pop()]
        while queue:
            current = queue.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    triangles = mesh.vertices[mesh.faces]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    edge_total = len(edge_counts)
    return {
        "vertex_count_welded": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "edge_count": int(edge_total),
        "boundary_edge_count": int(sum(value == 1 for value in edge_counts.values())),
        "nonmanifold_edge_count": int(sum(value > 2 for value in edge_counts.values())),
        "connected_components": int(components),
        "euler_characteristic": int(len(mesh.vertices) - edge_total + len(mesh.faces)),
        "surface_area": float(double_area.sum() * 0.5),
    }


def mesh_face_graph(
    asset: MeshAsset, weld_decimals: int | None = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create native face features, shared-edge neighbors, and a topology vector.

    Face features are center(3), normal(3), sorted edge lengths(3), log area(1),
    and cosine dihedral/boundary indicators(3). Coordinates and lengths are
    normalized by the mesh bounding-box diagonal.
    """
    mesh = compact_mesh(asset, weld_decimals=weld_decimals)
    vertices, faces = mesh.vertices, mesh.faces
    minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
    center = (minimum + maximum) * 0.5
    scale = max(float(np.linalg.norm(maximum - minimum)), 1e-12)
    normalized_vertices = (vertices - center) / scale
    triangles = normalized_vertices[faces]
    centers = triangles.mean(axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    face_normals = cross / np.maximum(double_area[:, None], 1e-12)
    edge_lengths = np.sort(
        np.stack(
            [
                np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
                np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
                np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            ],
            axis=1,
        ),
        axis=1,
    )

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    face_edges: list[list[tuple[int, int]]] = []
    for face_index, face in enumerate(faces):
        edges = []
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            edges.append(edge)
            edge_faces[edge].append(face_index)
        face_edges.append(edges)
    neighbors = np.empty((len(faces), 3), dtype=np.int64)
    dihedral = np.empty((len(faces), 3), dtype=np.float64)
    for face_index, edges in enumerate(face_edges):
        for edge_index, edge in enumerate(edges):
            other = next((value for value in edge_faces[edge] if value != face_index), face_index)
            neighbors[face_index, edge_index] = other
            dihedral[face_index, edge_index] = (
                -1.0 if other == face_index else np.dot(face_normals[face_index], face_normals[other])
            )
    features = np.concatenate(
        [centers, face_normals, edge_lengths, np.log1p(0.5 * double_area)[:, None], dihedral],
        axis=1,
    ).astype(np.float32)

    stats = topology_stats(mesh)
    edge_count = max(stats["edge_count"], 1)
    face_count = max(stats["face_count"], 1)
    topology = np.asarray(
        [
            np.log1p(stats["vertex_count_welded"]) / 10.0,
            np.log1p(stats["face_count"]) / 10.0,
            np.log1p(stats["edge_count"]) / 10.0,
            stats["boundary_edge_count"] / edge_count,
            stats["nonmanifold_edge_count"] / edge_count,
            np.log1p(stats["connected_components"]) / 5.0,
            stats["euler_characteristic"] / face_count,
            np.log1p(stats["surface_area"] / (scale * scale)) / 5.0,
        ],
        dtype=np.float32,
    )
    return features, neighbors, topology


def apply_mesh_attack(asset: MeshAsset, attack: str, severity: float, seed: int) -> MeshAsset:
    if attack == "qem":
        return qem_simplify(asset, severity)
    if attack == "connected_crop":
        return connected_crop(asset, 1.0 - severity, seed)
    if attack == "hole":
        return punch_hole(asset, severity, seed)
    if attack == "normal_flip":
        return flip_face_winding(asset, severity, seed)
    if attack == "retriangulate":
        return centroid_retriangulate(asset, severity, seed)
    raise ValueError(f"Unknown mesh attack: {attack}")
