"""Deterministic local mesh-patch descriptors used by no-reference QA."""

from __future__ import annotations

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
