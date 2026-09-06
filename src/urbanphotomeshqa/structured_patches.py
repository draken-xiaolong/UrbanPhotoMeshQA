"""Development candidate for bounded, normal-aware fixed-slot surface patches.

Unassignable fragments retain component IDs and face_patch=-1. They are never
silently bridged to make the layout pass admission. Defaults require development
calibration and later Val comparison; this does not replace the existing layout.
"""
from __future__ import annotations

import heapq
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from .patches import _components, _descriptor, _geometric_face_adjacency, _original_face_graph


def structured_patch_layout(mesh, patch_count=16, *, bridge_distance=0.025,
                            bridge_span=0.35, bridge_normal_cos=0.5,
                            normal_penalty=2.0):
    """Return CSR face membership in original face order and explicit validity.

    Distance and span bounds are fractions of the asset bounding-box diagonal.
    Virtual links use nearby face centers and the span of both endpoint faces,
    not an unrestricted component MST. Opposing endpoint normals cannot link.
    """
    if patch_count < 1 or bridge_distance < 0 or bridge_span < 0 or normal_penalty < 0:
        raise ValueError("Invalid layout bounds")
    if not -1 <= bridge_normal_cos <= 1:
        raise ValueError("Invalid normal cosine")
    vertices = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    if len(faces) == 0 or not np.isfinite(vertices).all():
        raise ValueError("Nonempty finite mesh required")
    triangles = vertices[faces]
    # Canonical geometric ordering makes tie breaks independent of file indices.
    keys = []
    for tri in triangles:
        keys.append(tuple(sorted(tuple(p) for p in tri)))
    order = np.asarray(sorted(range(len(faces)), key=lambda i: keys[i]), np.int64)
    canonical = SimpleNamespace(vertices=vertices, faces=faces[order])
    scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
    tri = (triangles[order] - vertices.min(axis=0)) / scale
    centers = tri.mean(axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    valid = area > 1e-14
    normals = cross / np.maximum(2 * area[:, None], 1e-30)
    real = _geometric_face_adjacency(canonical)
    real = [[j for j in neighbors if valid[i] and valid[j]]
            for i, neighbors in enumerate(real)]
    components = _components(real)
    component_id = np.empty(len(faces), np.int64)
    for i, component in enumerate(components):
        component_id[component] = i
    adjacency = [set(row) for row in real]
    bridges = []
    # A bounded candidate search is intentionally conservative. Missing a bridge
    # leaves an identifiable fragment instead of relaxing geometric constraints.
    tree = cKDTree(centers)
    distance, near = tree.query(centers, k=min(12, len(faces)))
    for i in range(len(faces)):
        for d, j in zip(np.atleast_1d(distance[i]), np.atleast_1d(near[i])):
            j = int(j)
            if j <= i or not (valid[i] and valid[j]) or component_id[i] == component_id[j]:
                continue
            span = float(np.linalg.norm(np.ptp(np.concatenate([tri[i], tri[j]]), axis=0)))
            cosine = float(normals[i] @ normals[j])
            if d <= bridge_distance and span <= bridge_span and cosine >= bridge_normal_cos:
                adjacency[i].add(j); adjacency[j].add(i)
                bridges.append((i, j, float(d), span, cosine))
    adjacency = [sorted(row) for row in adjacency]
    groups = [g[valid[g]] for g in _components(adjacency)]
    groups = [g for g in groups if len(g)]
    # If more isolated groups than slots exist, retain the largest ones. Record
    # the remaining faces separately and fail full-coverage suitability.
    groups.sort(key=lambda g: (-float(area[g].sum()), int(g.min())))
    selected = groups[:patch_count]
    allocation = np.ones(len(selected), np.int64)
    active = min(patch_count, sum(len(g) for g in selected))
    while allocation.sum() < active:
        candidates = [i for i, g in enumerate(selected) if allocation[i] < len(g)]
        i = max(candidates, key=lambda i: (area[selected[i]].sum() / allocation[i], -i))
        allocation[i] += 1

    def cost(i, j):
        return max(float(np.linalg.norm(centers[i] - centers[j])), 1e-12) * (
            1 + normal_penalty * (1 - float(np.clip(normals[i] @ normals[j], -1, 1))))

    def distances(seeds):
        values = np.full(len(faces), np.inf); queue = []
        for seed in seeds:
            values[seed] = 0; heapq.heappush(queue, (0., int(seed)))
        while queue:
            d, i = heapq.heappop(queue)
            if d != values[i]:
                continue
            for j in adjacency[i]:
                candidate = d + cost(i, j)
                if candidate < values[j]:
                    values[j] = candidate; heapq.heappush(queue, (candidate, j))
        return values

    seeds = []
    for group, count in zip(selected, allocation):
        center = np.average(centers[group], axis=0, weights=area[group])
        chosen = [int(group[np.argmax(np.sum((centers[group] - center) ** 2, axis=1))])]
        for _ in range(1, int(count)):
            d = distances(chosen)
            remaining = group[~np.isin(group, chosen)]
            chosen.append(int(remaining[np.argmax(d[remaining])]))
        seeds.extend(chosen)
    labels = np.full(len(faces), -1, np.int64)
    totals = np.zeros(active); frontiers = [[] for _ in seeds]
    for label, seed in enumerate(seeds):
        labels[seed] = label; totals[label] = area[seed]
    for label, seed in enumerate(seeds):
        for j in adjacency[seed]:
            if labels[j] < 0:
                heapq.heappush(frontiers[label], (cost(seed, j), j))
    while True:
        available = []
        for label, queue in enumerate(frontiers):
            while queue and labels[queue[0][1]] >= 0:
                heapq.heappop(queue)
            if queue:
                available.append(label)
        if not available:
            break
        label = min(available, key=lambda i: (totals[i], i))
        d, i = heapq.heappop(frontiers[label]); labels[i] = label; totals[label] += area[i]
        for j in adjacency[i]:
            if labels[j] < 0:
                heapq.heappush(frontiers[label], (d + cost(i, j), j))
    original_labels = np.empty_like(labels); original_labels[order] = labels
    original_components = np.empty_like(labels); original_components[order] = component_id
    features, neighbors = _original_face_graph(canonical)
    descriptors = np.zeros((patch_count, 58), np.float32)
    areas = np.zeros(patch_count); patch_centers = np.zeros((patch_count, 3))
    members = []; offsets = [0]; real_counts = np.zeros(patch_count, np.int64)
    spans = np.zeros(patch_count)
    for label in range(patch_count):
        local = np.flatnonzero(labels == label)
        members.extend(order[local].tolist()); offsets.append(len(members))
        if len(local):
            descriptors[label] = _descriptor(features, neighbors, local)
            areas[label] = area[local].sum() * scale ** 2
            patch_centers[label] = np.average(triangles[order[local]].mean(axis=1), axis=0, weights=area[local])
            spans[label] = np.linalg.norm(np.ptp(tri[local].reshape(-1, 3), axis=0))
            mapping = {int(f): j for j, f in enumerate(local)}
            real_counts[label] = len(_components([[mapping[j] for j in real[i] if j in mapping] for i in local]))
    valid_original = np.empty_like(valid); valid_original[order] = valid
    return dict(schema=np.asarray("structured16_development_v1"), descriptors=descriptors,
                patch_mask=areas > 0, face_patch=original_labels,
                patch_offsets=np.asarray(offsets, np.int64), patch_face_indices=np.asarray(members, np.int64),
                patch_area=areas, patch_center=patch_centers, face_valid=valid_original,
                real_component_id=original_components, patch_real_component_count=real_counts,
                patch_span_fraction=spans,
                virtual_bridge_faces=np.asarray([(order[i], order[j]) for i,j,*_ in bridges], np.int64).reshape(-1,2),
                virtual_bridge_metrics=np.asarray([b[2:] for b in bridges], np.float64).reshape(-1,3),
                unassigned_face_indices=np.flatnonzero(original_labels < 0),
                full_valid_coverage=np.asarray(np.all(original_labels[valid_original] >= 0)),
                formal_admitted=np.asarray(False))
