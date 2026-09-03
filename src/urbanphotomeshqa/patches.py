"""Deterministic local mesh-patch descriptors used by no-reference QA."""

from __future__ import annotations

import heapq
import numpy as np

from .mesh_attacks import mesh_face_graph


def farthest_centers(centers: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(centers))
    chosen = [int(np.argmax(np.sum((centers - centers.mean(0)) ** 2, axis=1)))]
    minimum = np.sum((centers - centers[chosen[0]]) ** 2, axis=1)
    for _ in range(1, count):
        chosen.append(int(np.argmax(minimum)))
        minimum = np.minimum(minimum, np.sum((centers - centers[chosen[-1]]) ** 2, axis=1))
    return np.asarray(chosen, dtype=np.int64)


def patch_layout(mesh, patch_count: int, neighbors_per_patch: int):
    """Return descriptors and the exact face membership of every patch."""
    features, neighbors, _ = mesh_face_graph(mesh)
    centers = features[:, :3]
    seeds = farthest_centers(centers, patch_count)
    output = np.zeros((patch_count, 58), dtype=np.float32)
    mask = np.zeros(patch_count, dtype=bool)
    face_indices = np.full((patch_count, neighbors_per_patch), -1, dtype=np.int64)
    face_mask = np.zeros((patch_count, neighbors_per_patch), dtype=bool)
    for patch_index, seed in enumerate(seeds):
        distance = np.sum((centers - centers[seed]) ** 2, axis=1)
        local = np.argsort(distance)[: min(neighbors_per_patch, len(features))]
        values = features[local]
        aggregate = np.concatenate([values.mean(0), values.std(0), values.min(0), values.max(0)])
        normals = values[:, 3:6]
        normal_dispersion = 1.0 - float(np.linalg.norm(normals.mean(0)))
        dihedral = values[:, 10:13]
        boundary_fraction = float(np.mean(dihedral <= -0.999))
        sharp_fraction = float(np.mean((dihedral > -0.999) & (dihedral < 0.8)))
        local_set = set(local.tolist())
        degrees = [sum(int(other) != int(face) and int(other) in local_set for other in neighbors[face])
                   for face in local]
        degree_mean = float(np.mean(degrees) / 3.0)
        aspect = values[:, 8] / np.maximum(values[:, 6], 1e-8)
        extras = np.asarray([normal_dispersion, boundary_fraction, sharp_fraction,
                             degree_mean, float(aspect.mean()), float(aspect.std())], dtype=np.float32)
        output[patch_index] = np.concatenate([aggregate, extras])
        mask[patch_index] = True
        face_indices[patch_index, :len(local)] = local
        face_mask[patch_index, :len(local)] = True
    return output, mask, face_indices, face_mask


def patch_descriptors(mesh, patch_count: int, neighbors_per_patch: int):
    descriptors, mask, _, _ = patch_layout(mesh, patch_count, neighbors_per_patch)
    return descriptors, mask


def _geometric_face_adjacency(mesh, decimals: int = 8) -> list[list[int]]:
    """Shared-edge adjacency that preserves original face indices across UV seams."""
    vertices = np.round(np.asarray(mesh.vertices, np.float64), decimals=decimals)
    edge_faces = {}
    for face_index, face in enumerate(np.asarray(mesh.faces, np.int64)):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            left, right = tuple(vertices[a]), tuple(vertices[b])
            edge = tuple(sorted((left, right)))
            edge_faces.setdefault(edge, []).append(face_index)
    adjacency = [set() for _ in range(len(mesh.faces))]
    for attached in edge_faces.values():
        for face in attached:
            adjacency[face].update(other for other in attached if other != face)
    return [sorted(values) for values in adjacency]


def _original_face_graph(mesh, decimals: int = 8):
    """Face features/neighbors without compacting or renumbering glTF faces."""
    vertices = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    scale = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), 1e-12)
    normalized = (vertices - center) / scale
    triangles = normalized[faces]
    centers = triangles.mean(axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(double_area[:, None], 1e-12)
    lengths = np.sort(np.stack([
        np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
        np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
    ], axis=1), axis=1)
    rounded = np.round(vertices, decimals=decimals)
    edge_faces = {}; face_edges = []
    for face_index, face in enumerate(faces):
        edges = []
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((tuple(rounded[a]), tuple(rounded[b]))))
            edge_faces.setdefault(edge, []).append(face_index); edges.append(edge)
        face_edges.append(edges)
    neighbors = np.empty((len(faces), 3), np.int64); dihedral = np.empty((len(faces), 3), np.float64)
    for face_index, edges in enumerate(face_edges):
        for edge_index, edge in enumerate(edges):
            other = next((value for value in edge_faces[edge] if value != face_index), face_index)
            neighbors[face_index, edge_index] = other
            dihedral[face_index, edge_index] = (-1.0 if other == face_index
                                                 else np.dot(normals[face_index], normals[other]))
    features = np.concatenate([centers, normals, lengths, np.log1p(0.5 * double_area)[:, None],
                               dihedral], axis=1).astype(np.float32)
    return features, neighbors


def _components(adjacency: list[list[int]]) -> list[np.ndarray]:
    remaining = set(range(len(adjacency))); output = []
    while remaining:
        start = min(remaining); remaining.remove(start); stack = [start]; component = []
        while stack:
            face = stack.pop(); component.append(face)
            for neighbor in adjacency[face]:
                if neighbor in remaining:
                    remaining.remove(neighbor); stack.append(neighbor)
        output.append(np.asarray(sorted(component), np.int64))
    return output


def _augment_component_bridges(adjacency, centers, face_areas):
    """Connect separate object parts with a deterministic spatial MST.

    Official building assets legitimately contain many disconnected objects
    (windows, rails, ornaments).  The virtual edges keep a fixed patch budget
    possible while preserving all real shared-edge adjacency.  They are used
    only for partitioning and are explicitly audited in the sidecar.
    """
    real_components = _components(adjacency)
    if len(real_components) <= 1:
        return adjacency, real_components, 0
    component_centers = np.stack([
        np.average(centers[component], axis=0,
                   weights=np.maximum(face_areas[component], 1e-12))
        for component in real_components
    ])
    count = len(real_components)
    selected = np.zeros(count, bool); selected[0] = True
    minimum = np.linalg.norm(component_centers - component_centers[0], axis=1)
    parent = np.zeros(count, np.int64); minimum[0] = np.inf
    bridges = []
    for _ in range(1, count):
        candidates = np.flatnonzero(~selected)
        child = int(candidates[np.argmin(minimum[candidates])])
        source = int(parent[child]); selected[child] = True
        left = int(real_components[source][np.argmin(np.linalg.norm(
            centers[real_components[source]] - component_centers[child], axis=1))])
        right = int(real_components[child][np.argmin(np.linalg.norm(
            centers[real_components[child]] - component_centers[source], axis=1))])
        bridges.append((left, right))
        distances = np.linalg.norm(component_centers - component_centers[child], axis=1)
        update = (~selected) & (distances < minimum)
        minimum[update] = distances[update]; parent[update] = child
    augmented = [set(values) for values in adjacency]
    for left, right in bridges:
        augmented[left].add(right); augmented[right].add(left)
    return [sorted(values) for values in augmented], real_components, len(bridges)


def _distances(adjacency, centers, seeds):
    distance = np.full(len(adjacency), np.inf, np.float64); queue = []
    for seed in seeds:
        distance[seed] = 0.0; heapq.heappush(queue, (0.0, int(seed)))
    while queue:
        current_distance, face = heapq.heappop(queue)
        if current_distance != distance[face]:
            continue
        for neighbor in adjacency[face]:
            candidate = current_distance + max(float(np.linalg.norm(centers[face] - centers[neighbor])), 1e-9)
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate; heapq.heappush(queue, (candidate, neighbor))
    return distance


def _descriptor(features, neighbors, local):
    values = features[local]
    aggregate = np.concatenate([values.mean(0), values.std(0), values.min(0), values.max(0)])
    normals = values[:, 3:6]
    normal_dispersion = 1.0 - float(np.linalg.norm(normals.mean(0)))
    dihedral = values[:, 10:13]
    boundary_fraction = float(np.mean(dihedral <= -0.999))
    sharp_fraction = float(np.mean((dihedral > -0.999) & (dihedral < 0.8)))
    local_set = set(local.tolist())
    degrees = [sum(int(other) != int(face) and int(other) in local_set for other in neighbors[face])
               for face in local]
    degree_mean = float(np.mean(degrees) / 3.0)
    aspect = values[:, 8] / np.maximum(values[:, 6], 1e-8)
    extras = np.asarray([normal_dispersion, boundary_fraction, sharp_fraction,
                         degree_mean, float(aspect.mean()), float(aspect.std())], dtype=np.float32)
    return np.concatenate([aggregate, extras]).astype(np.float32)


def _balanced_region_grow(adjacency, centers, face_areas, seeds):
    """Grow connected regions while prioritizing the least-filled area quota."""
    patch_count = len(seeds); face_count = len(adjacency)
    labels = np.full(face_count, -1, np.int64)
    accumulated = np.zeros(patch_count, np.float64)
    frontiers = [[] for _ in range(patch_count)]
    for label, seed in enumerate(seeds):
        labels[seed] = label; accumulated[label] = face_areas[seed]
    for label, seed in enumerate(seeds):
        for neighbor in adjacency[seed]:
            if labels[neighbor] < 0:
                distance = max(float(np.linalg.norm(centers[seed] - centers[neighbor])), 1e-9)
                heapq.heappush(frontiers[label], (distance, int(neighbor)))
    target = max(float(face_areas.sum()) / patch_count, 1e-12)
    remaining = face_count - patch_count
    while remaining:
        available = []
        for label, frontier in enumerate(frontiers):
            while frontier and labels[frontier[0][1]] >= 0:
                heapq.heappop(frontier)
            if frontier:
                available.append(label)
        if not available:
            raise ValueError("Connected region growth left unreachable faces")
        label = min(available, key=lambda value: (accumulated[value] / target, value))
        distance, face = heapq.heappop(frontiers[label])
        if labels[face] >= 0:
            continue
        labels[face] = label; accumulated[label] += face_areas[face]; remaining -= 1
        for neighbor in adjacency[face]:
            if labels[neighbor] < 0:
                step = max(float(np.linalg.norm(centers[face] - centers[neighbor])), 1e-9)
                heapq.heappush(frontiers[label], (distance + step, int(neighbor)))
    return labels


def topological_patch_layout(mesh, patch_count: int = 16) -> dict[str, np.ndarray]:
    """Partition every face once into deterministic spatial-geodesic patches.

    Unlike :func:`patch_layout`, this v2 representation has full, non-overlapping
    coverage and CSR membership suitable for exact Face↔UV supervision.  Real
    shared-edge topology is augmented by an audited spatial MST for disconnected
    building components so that a fixed patch budget remains practical.
    """
    features, legacy_neighbors = _original_face_graph(mesh)
    face_count = len(mesh.faces)
    adjacency = _geometric_face_adjacency(mesh)
    if face_count == 0:
        raise ValueError("Cannot partition an empty mesh")
    requested = max(int(patch_count), 1)
    triangles = np.asarray(mesh.vertices, np.float64)[np.asarray(mesh.faces, np.int64)]
    centers_world = triangles.mean(axis=1)
    face_areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                               triangles[:, 2] - triangles[:, 0]), axis=1)
    adjacency, real_components, virtual_bridge_count = _augment_component_bridges(
        adjacency, centers_world, face_areas)
    components = _components(adjacency)
    active = min(face_count, requested)
    allocation = np.ones(len(components), np.int64)
    component_areas = np.asarray([face_areas[value].sum() for value in components])
    while allocation.sum() < active:
        candidates = [index for index, component in enumerate(components)
                      if allocation[index] < len(component)]
        chosen = max(candidates, key=lambda index: (component_areas[index] / allocation[index], -index))
        allocation[chosen] += 1
    seeds = []
    for component, count in zip(components, allocation):
        weighted_center = np.average(centers_world[component], axis=0,
                                     weights=np.maximum(face_areas[component], 1e-12))
        chosen = [int(component[np.argmax(np.sum((centers_world[component] - weighted_center) ** 2, axis=1))])]
        for _ in range(1, int(count)):
            distance = _distances(adjacency, centers_world, chosen)
            chosen.append(int(component[np.argmax(distance[component])]))
        seeds.extend(chosen)
    labels = _balanced_region_grow(adjacency, centers_world, face_areas, seeds)
    if np.any(labels < 0):
        raise ValueError("Topological partition left unassigned faces")
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=active)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    descriptors = np.zeros((active, 58), np.float32)
    patch_mask = np.ones(active, bool)
    patch_area = np.zeros(active, np.float64)
    patch_center = np.zeros((active, 3), np.float64)
    for patch in range(active):
        local = order[offsets[patch]:offsets[patch + 1]]
        descriptors[patch] = _descriptor(features, legacy_neighbors, local)
        patch_area[patch] = face_areas[local].sum()
        patch_center[patch] = np.average(centers_world[local], axis=0,
                                         weights=np.maximum(face_areas[local], 1e-12))
        visited = {int(local[0])}; stack = [int(local[0])]; local_set = set(local.tolist())
        while stack:
            face = stack.pop()
            for neighbor in adjacency[face]:
                if neighbor in local_set and neighbor not in visited:
                    visited.add(neighbor); stack.append(neighbor)
        if len(visited) != len(local):
            raise ValueError(f"Patch {patch} is not edge-connected")
    return {"descriptors": descriptors, "patch_mask": patch_mask,
            "face_patch": labels, "patch_offsets": offsets,
            "patch_face_indices": order.astype(np.int64), "patch_area": patch_area.astype(np.float32),
            "patch_center": patch_center.astype(np.float64),
            "connected_components": np.asarray(len(real_components), np.int64),
            "virtual_bridge_count": np.asarray(virtual_bridge_count, np.int64),
            "requested_patch_count": np.asarray(requested, np.int64)}
