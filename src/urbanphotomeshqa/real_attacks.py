"""File-native geometry and source-texture degradations for quality assessment."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image, ImageFilter

from .gltf import MeshAsset
from .mesh_attacks import connected_face_patch, face_adjacency, recompute_vertex_normals
from .texture import _compact_textured, _topology_faces


def _multi_component_patch(faces: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Grow one or more connected patches until the exact requested count."""
    adjacency = face_adjacency(faces)
    selected: set[int] = set()
    target = min(max(1, int(count)), len(faces))
    while len(selected) < target:
        candidates = np.asarray(sorted(set(range(len(faces))) - selected), dtype=np.int64)
        seed = int(rng.choice(candidates))
        queue = [seed]
        queued = {seed}
        while queue and len(selected) < target:
            current = queue.pop(0)
            if current in selected:
                continue
            selected.add(current)
            neighbors = np.asarray(adjacency[current], dtype=np.int64)
            rng.shuffle(neighbors)
            for neighbor in neighbors.tolist():
                if neighbor not in selected and neighbor not in queued:
                    queue.append(neighbor)
                    queued.add(neighbor)
    return np.asarray(sorted(selected), dtype=np.int64)


def geometry_hole(asset: MeshAsset, removed_fraction: float, seed: int) -> MeshAsset:
    mesh = _compact_textured(asset)
    count = min(max(1, int(round(len(mesh.faces) * removed_fraction))), len(mesh.faces) - 1)
    selected = _multi_component_patch(_topology_faces(mesh), count, np.random.default_rng(seed))
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[selected] = False
    retained = np.flatnonzero(keep)
    result = MeshAsset(
        vertices=mesh.vertices,
        faces=mesh.faces[retained],
        normals=mesh.normals,
        face_materials=mesh.face_materials[retained],
        metadata={**mesh.metadata, "attack": "geometry_hole", "removed_fraction": removed_fraction},
        texcoords=mesh.texcoords,
    )
    return _compact_textured(result)


def geometry_noise_spike(
    asset: MeshAsset,
    diagonal_fraction: float,
    affected_face_fraction: float,
    seed: int,
) -> MeshAsset:
    rng = np.random.default_rng(seed)
    topology_faces = _topology_faces(asset)
    count = max(1, int(round(len(asset.faces) * affected_face_fraction)))
    selected_faces = connected_face_patch(topology_faces, count, rng)
    selected_vertices = np.unique(asset.faces[selected_faces].reshape(-1))
    diagonal = float(np.linalg.norm(np.ptp(asset.vertices, axis=0)))
    displacement = np.zeros(len(asset.vertices), dtype=np.float64)
    displacement[selected_vertices] = rng.normal(0.0, diagonal_fraction * diagonal / 3.0, len(selected_vertices))
    if len(selected_vertices):
        spike = int(rng.choice(selected_vertices))
        displacement[spike] = diagonal_fraction * diagonal
    vertices = np.asarray(asset.vertices, dtype=np.float64) + asset.normals * displacement[:, None]
    return replace(
        asset,
        vertices=vertices,
        normals=recompute_vertex_normals(vertices, asset.faces),
        metadata={
            **asset.metadata,
            "attack": "geometry_noise_spike",
            "diagonal_fraction": float(diagonal_fraction),
            "affected_face_fraction": float(affected_face_fraction),
        },
    )


def qem_simplify_textured(asset: MeshAsset, retained_fraction: float) -> MeshAsset:
    import fast_simplification
    from scipy.spatial import cKDTree

    if asset.texcoords is None:
        raise ValueError("QEM texture reprojection requires TEXCOORD_0")
    # glTF render vertices are split at UV seams. Weld positions only for the
    # simplifier, then project the original atlas/material back per output face.
    rounded, inverse = np.unique(np.round(asset.vertices, 8), axis=0, return_inverse=True)
    welded_faces = inverse[np.asarray(asset.faces, dtype=np.int64)]
    valid = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 0] != welded_faces[:, 2])
    )
    welded_faces = welded_faces[valid].astype(np.int32)
    target = max(4, int(round(len(welded_faces) * retained_fraction)))
    simplified_vertices, simplified_faces = fast_simplification.simplify(
        rounded.astype(np.float64), welded_faces, target_count=target, preserve_border=False, agg=10.0
    )
    simplified_vertices = np.asarray(simplified_vertices, dtype=np.float64)
    simplified_faces = np.asarray(simplified_faces, dtype=np.int64)
    if len(simplified_faces) >= len(welded_faces):
        raise RuntimeError("QEM simplification produced no face reduction")

    original_triangles = np.asarray(asset.vertices, dtype=np.float64)[asset.faces]
    original_centroids = original_triangles.mean(axis=1)
    output_triangles = simplified_vertices[simplified_faces]
    _, nearest_faces = cKDTree(original_centroids).query(output_triangles.mean(axis=1), k=1)
    projected_uv = np.empty((len(output_triangles), 3, 2), dtype=np.float64)
    original_uv = np.asarray(asset.texcoords, dtype=np.float64)[asset.faces]
    for index, (triangle, original_index) in enumerate(zip(output_triangles, nearest_faces)):
        reference = original_triangles[int(original_index)]
        basis = np.column_stack((reference[1] - reference[0], reference[2] - reference[0]))
        coordinates = np.linalg.lstsq(basis, (triangle - reference[0]).T, rcond=None)[0].T
        barycentric = np.column_stack((1.0 - coordinates.sum(axis=1), coordinates))
        barycentric = np.clip(barycentric, 0.0, 1.0)
        barycentric /= np.maximum(barycentric.sum(axis=1, keepdims=True), 1e-12)
        projected_uv[index] = barycentric @ original_uv[int(original_index)]

    # Duplicate corners so material boundaries and UV seams remain exact.
    vertices = output_triangles.reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return MeshAsset(
        vertices=vertices,
        faces=faces,
        normals=recompute_vertex_normals(vertices, faces),
        face_materials=np.asarray(asset.face_materials, dtype=np.int32)[nearest_faces],
        metadata={**asset.metadata, "attack": "mesh_simplification_qem", "retained_fraction": retained_fraction},
        texcoords=projected_uv.reshape(-1, 2),
    )


def texture_detail_loss(image: Image.Image, subtype: str, value: float) -> Image.Image:
    image = image.convert("RGBA")
    if subtype == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if subtype == "texture_downsample":
        width = max(1, int(round(image.width * value)))
        height = max(1, int(round(image.height * value)))
        return image.resize((width, height), Image.Resampling.LANCZOS)
    raise ValueError(subtype)


def _importance_center(
    image: Image.Image,
    importance_mask: np.ndarray | None,
    seed: int,
    mode: str,
) -> tuple[float, float] | None:
    if importance_mask is None or not np.asarray(importance_mask, bool).any():
        return None
    mask = np.asarray(importance_mask, bool)
    thumbnail = np.asarray(
        image.convert("RGB").resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    if mode == "missing":
        signal = np.mean(np.abs(thumbnail - 128.0), axis=2)
    elif mode == "misalignment":
        signal = np.mean(np.abs(thumbnail - np.roll(thumbnail, 1, axis=1)), axis=2)
    else:
        raise ValueError(mode)
    scores = np.where(mask, signal, -np.inf).reshape(-1)
    maximum = float(np.max(scores))
    if not np.isfinite(maximum):
        return None
    # Select among near-equal maxima rather than sampling the whole UV mask:
    # large atlas padding otherwise dominates sparse facade edges.
    candidates = np.flatnonzero(scores >= maximum - 1e-6)
    selected = int(np.random.default_rng(seed).choice(candidates))
    row, column = divmod(selected, mask.shape[1])
    return ((row + 0.5) / mask.shape[0], (column + 0.5) / mask.shape[1])


def _window_from_center(
    image_height: int,
    image_width: int,
    region_height: int,
    region_width: int,
    center: tuple[float, float],
) -> tuple[int, int]:
    center_y, center_x = center
    top = int(round(center_y * image_height - region_height * 0.5))
    left = int(round(center_x * image_width - region_width * 0.5))
    return (
        int(np.clip(top, 0, max(0, image_height - region_height))),
        int(np.clip(left, 0, max(0, image_width - region_width))),
    )


def texture_region_missing(
    image: Image.Image,
    fraction: float,
    seed: int,
    importance_mask: np.ndarray | None = None,
) -> tuple[Image.Image, float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    target = max(1, int(round(fraction * array.shape[0] * array.shape[1])))
    aspect = float(rng.uniform(0.65, 1.55))
    height = min(array.shape[0], max(1, int(round(np.sqrt(target / aspect)))))
    width = min(array.shape[1], max(1, int(round(target / height))))
    center = _importance_center(image, importance_mask, seed, "missing")
    if center is None:
        top = int(rng.integers(0, max(1, array.shape[0] - height + 1)))
        left = int(rng.integers(0, max(1, array.shape[1] - width + 1)))
    else:
        top, left = _window_from_center(
            array.shape[0], array.shape[1], height, width, center
        )
    array[top : top + height, left : left + width, :3] = 128
    array[top : top + height, left : left + width, 3] = 255
    actual = (height * width) / (array.shape[0] * array.shape[1])
    return Image.fromarray(array), float(actual)


def texture_misalignment(
    image: Image.Image,
    shift_fraction: float,
    ghost_alpha: float,
    seed: int,
    importance_mask: np.ndarray | None = None,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    height, width = array.shape[:2]
    region_width = max(2, int(round(width * 0.55)))
    region_height = max(2, int(round(height * 0.45)))
    center = _importance_center(image, importance_mask, seed, "misalignment")
    if center is not None:
        top, left = _window_from_center(height, width, region_height, region_width, center)
    else:
        # Legacy fallback for assets without a valid UV-usage mask.
        candidates = []
        for top_fraction in np.linspace(0.0, 1.0, 5):
            for left_fraction in np.linspace(0.0, 1.0, 5):
                top = int(round(top_fraction * max(0, height - region_height)))
                left = int(round(left_fraction * max(0, width - region_width)))
                sample = array[top : top + region_height : 8, left : left + region_width : 8, :3]
                score = float(np.var(sample.astype(np.float32))) + float(rng.uniform(0.0, 1e-6))
                candidates.append((score, top, left))
        _, top, left = max(candidates)
    region = array[top : top + region_height, left : left + region_width].copy()
    shift = max(1, int(round(width * shift_fraction)))
    shifted = np.roll(region, shift=shift, axis=1)
    blended = np.rint((1.0 - ghost_alpha) * region.astype(np.float32) + ghost_alpha * shifted.astype(np.float32))
    array[top : top + region_height, left : left + region_width] = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(array)
