from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .gltf import MeshAsset
from .mesh_attacks import connected_face_patch, recompute_vertex_normals


def _compact_textured(asset: MeshAsset) -> MeshAsset:
    if asset.texcoords is None:
        raise ValueError("Textured operation requires TEXCOORD_0")
    faces = np.asarray(asset.faces, dtype=np.int64)
    used, remapped = np.unique(faces.reshape(-1), return_inverse=True)
    vertices = np.asarray(asset.vertices, dtype=np.float64)[used]
    faces = remapped.reshape(-1, 3)
    return MeshAsset(
        vertices=vertices,
        faces=faces,
        normals=recompute_vertex_normals(vertices, faces),
        face_materials=np.asarray(asset.face_materials, dtype=np.int32).copy(),
        metadata=dict(asset.metadata),
        texcoords=np.asarray(asset.texcoords, dtype=np.float64)[used],
    )


def _textured_subset(asset: MeshAsset, indices: np.ndarray, attack: str, severity: float) -> MeshAsset:
    return _compact_textured(MeshAsset(
        vertices=asset.vertices,
        faces=asset.faces[indices],
        normals=asset.normals,
        face_materials=asset.face_materials[indices],
        metadata={**asset.metadata, "attack": attack, "severity": float(severity)},
        texcoords=asset.texcoords,
    ))


def _topology_faces(asset: MeshAsset, decimals: int = 8) -> np.ndarray:
    """Weld positions only for adjacency, without modifying render vertices or UV seams."""
    _, inverse = np.unique(np.round(asset.vertices, decimals), axis=0, return_inverse=True)
    return inverse[np.asarray(asset.faces, dtype=np.int64)]


def apply_uv_preserving_attack(asset: MeshAsset, attack: str, severity: float, seed: int) -> MeshAsset:
    """Apply only attacks whose vertex/UV correspondence is exact.

    QEM is intentionally excluded: common simplifiers return new vertices but
    no trustworthy map back to the original UV atlas.
    """
    mesh = _compact_textured(asset)
    rng = np.random.default_rng(seed)
    topology_faces = _topology_faces(mesh)
    if attack == "connected_crop":
        keep = max(1, int(round(len(mesh.faces) * (1.0 - severity))))
        selected = connected_face_patch(topology_faces, keep, rng)
        return _textured_subset(mesh, selected, attack, severity)
    if attack == "hole":
        if len(mesh.faces) < 2:
            raise ValueError("Cannot remove a hole from a one-face mesh")
        remove = min(max(1, int(round(len(mesh.faces) * severity))), len(mesh.faces) - 1)
        selected = connected_face_patch(topology_faces, remove, rng)
        keep = np.ones(len(mesh.faces), dtype=bool); keep[selected] = False
        return _textured_subset(mesh, np.flatnonzero(keep), attack, severity)
    if attack == "normal_flip":
        count = max(1, int(round(len(mesh.faces) * severity)))
        selected = connected_face_patch(topology_faces, count, rng)
        faces = mesh.faces.copy(); faces[selected] = faces[selected][:, [0, 2, 1]]
        return replace(
            mesh, faces=faces, normals=recompute_vertex_normals(mesh.vertices, faces),
            metadata={**mesh.metadata, "attack": attack, "severity": float(severity)},
        )
    if attack == "retriangulate":
        count = max(1, int(round(len(mesh.faces) * severity)))
        selected = set(connected_face_patch(topology_faces, count, rng).tolist())
        vertices, texcoords = mesh.vertices.tolist(), mesh.texcoords.tolist()
        faces, materials = [], []
        for index, face in enumerate(mesh.faces):
            material = int(mesh.face_materials[index])
            if index not in selected:
                faces.append(face.tolist()); materials.append(material); continue
            center = len(vertices)
            vertices.append(mesh.vertices[face].mean(axis=0).tolist())
            texcoords.append(mesh.texcoords[face].mean(axis=0).tolist())
            a, b, c = face.tolist()
            faces.extend([[a, b, center], [b, c, center], [c, a, center]])
            materials.extend([material] * 3)
        vertices_array = np.asarray(vertices, dtype=np.float64)
        faces_array = np.asarray(faces, dtype=np.int64)
        return MeshAsset(
            vertices=vertices_array, faces=faces_array,
            normals=recompute_vertex_normals(vertices_array, faces_array),
            face_materials=np.asarray(materials, dtype=np.int32),
            metadata={**mesh.metadata, "attack": attack, "severity": float(severity)},
            texcoords=np.asarray(texcoords, dtype=np.float64),
        )
    if attack == "qem":
        raise NotImplementedError("QEM requires UV reprojection and is excluded from the leakage-free texture protocol")
    raise ValueError(f"Unknown UV-preserving attack: {attack}")


def _camera_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = direction / np.linalg.norm(direction)
    forward = -camera
    up_hint = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up_hint))) > 0.95:
        up_hint = np.asarray([0.0, 1.0, 0.0])
    right = np.cross(forward, up_hint); right /= np.linalg.norm(right)
    up = np.cross(right, forward); up /= np.linalg.norm(up)
    return right, up, camera


def _load_textures(asset: MeshAsset) -> list[np.ndarray | None]:
    embedded = asset.metadata.get("material_texture_arrays")
    if embedded is not None:
        return [None if value is None else np.asarray(value, dtype=np.uint8) for value in embedded]
    result = []
    for value in asset.metadata.get("material_texture_paths", []):
        if value is None or not Path(value).exists():
            result.append(None)
            continue
        result.append(_load_texture_path(str(value)))
    return result


@lru_cache(maxsize=128)
def _load_texture_path(value: str) -> np.ndarray:
    return np.asarray(Image.open(value).convert("RGBA"), dtype=np.uint8)


def _wrap_texture_coordinate(values: np.ndarray, mode: int) -> np.ndarray:
    """Apply the glTF sampler wrap mode to one normalized UV component."""
    if mode == 33071:  # CLAMP_TO_EDGE
        return np.clip(values, 0.0, 1.0)
    if mode == 33648:  # MIRRORED_REPEAT
        period = np.mod(values, 2.0)
        return np.where(period <= 1.0, period, 2.0 - period)
    return values - np.floor(values)  # REPEAT (the glTF default)


def _material_render_profile(asset: MeshAsset, material: int) -> dict:
    profiles = asset.metadata.get("material_profiles", [])
    profile = profiles[material] if 0 <= material < len(profiles) else {}
    pbr = profile.get("pbrMetallicRoughness") or {}
    factor = np.asarray(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]), dtype=np.float64)
    if factor.shape != (4,):
        factor = np.ones(4, dtype=np.float64)
    sampler = profile.get("baseColorSampler") or {}
    return {
        "factor": np.clip(factor, 0.0, 1.0),
        "alpha_mode": str(profile.get("alphaMode", "OPAQUE")).upper(),
        "alpha_cutoff": float(profile.get("alphaCutoff") or 0.5),
        "wrap_s": int(sampler.get("wrapS", 10497)),
        "wrap_t": int(sampler.get("wrapT", 10497)),
    }


def render_textured_view(
    asset: MeshAsset,
    direction: tuple[float, float, float] = (1.0, 1.0, 0.7),
    size: int = 192,
    background: tuple[int, int, int] = (245, 245, 245),
) -> np.ndarray:
    """Small deterministic orthographic CPU rasterizer for independent RGB views."""
    if asset.texcoords is None or len(asset.texcoords) != len(asset.vertices):
        raise ValueError("Mesh does not contain vertex-aligned TEXCOORD_0")
    vertices = np.asarray(asset.vertices, dtype=np.float64)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    scale = max(float(np.max(vertices.max(axis=0) - vertices.min(axis=0))), 1e-12)
    points = (vertices - center) / scale
    right, up, camera = _camera_basis(np.asarray(direction, dtype=np.float64))
    projected = np.column_stack([points @ right, points @ up, points @ camera])
    xy = (projected[:, :2] * 0.82 + 0.5) * (size - 1)
    depth = projected[:, 2]
    image = np.empty((size, size, 3), dtype=np.uint8); image[:] = background
    zbuffer = np.full((size, size), -np.inf, dtype=np.float64)
    textures = _load_textures(asset)
    texcoords = np.asarray(asset.texcoords, dtype=np.float64)

    for face_index, face in enumerate(np.asarray(asset.faces, dtype=np.int64)):
        triangle = xy[face]
        minimum = np.maximum(np.floor(triangle.min(axis=0)).astype(int), 0)
        maximum = np.minimum(np.ceil(triangle.max(axis=0)).astype(int), size - 1)
        if np.any(maximum < minimum):
            continue
        x0, y0 = triangle[0]; x1, y1 = triangle[1]; x2, y2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denominator)) < 1e-10:
            continue
        xs, ys = np.meshgrid(
            np.arange(minimum[0], maximum[0] + 1), np.arange(minimum[1], maximum[1] + 1)
        )
        w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denominator
        w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        interpolated_depth = w0 * depth[face[0]] + w1 * depth[face[1]] + w2 * depth[face[2]]
        target_depth = zbuffer[ys, xs]
        visible = inside & (interpolated_depth > target_depth)
        if not np.any(visible):
            continue
        material = int(asset.face_materials[face_index])
        texture = textures[material] if 0 <= material < len(textures) else None
        if texture is None or not np.isfinite(texcoords[face]).all():
            colors = np.broadcast_to(np.asarray([180, 180, 180], dtype=np.uint8), (*xs.shape, 3))
            alpha = np.ones(xs.shape, dtype=np.float64)
        else:
            uv = w0[..., None] * texcoords[face[0]] + w1[..., None] * texcoords[face[1]] + w2[..., None] * texcoords[face[2]]
            uv = uv - np.floor(uv)
            tx = np.clip(np.rint(uv[..., 0] * (texture.shape[1] - 1)).astype(int), 0, texture.shape[1] - 1)
            ty = np.clip(np.rint((1.0 - uv[..., 1]) * (texture.shape[0] - 1)).astype(int), 0, texture.shape[0] - 1)
            sampled = texture[ty, tx]
            colors, alpha = sampled[..., :3], sampled[..., 3].astype(np.float64) / 255.0
        region = image[ys, xs]
        blended = np.rint(colors * alpha[..., None] + region * (1.0 - alpha[..., None])).astype(np.uint8)
        region[visible] = blended[visible]
        image[ys, xs] = region
        target_depth[visible] = interpolated_depth[visible]
        zbuffer[ys, xs] = target_depth
    return image


def render_textured_view_with_masks(
    asset: MeshAsset,
    direction: tuple[float, float, float] = (1.0, 1.0, 0.7),
    size: int = 192,
    background: tuple[int, int, int] = (245, 245, 245),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a material-aware RGB view and explicit visible-region masks.

    Returns ``(rgb, foreground_mask, textured_mask)``.  The masks are derived
    from rasterization rather than background color, so pale or missing texture
    pixels cannot be mistaken for background.  This is the v2 target renderer;
    :func:`render_textured_view` remains unchanged for v1 cache reproducibility.
    """
    if asset.texcoords is None or len(asset.texcoords) != len(asset.vertices):
        raise ValueError("Mesh does not contain vertex-aligned TEXCOORD_0")
    vertices = np.asarray(asset.vertices, dtype=np.float64)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    scale = max(float(np.max(vertices.max(axis=0) - vertices.min(axis=0))), 1e-12)
    points = (vertices - center) / scale
    right, up, camera = _camera_basis(np.asarray(direction, dtype=np.float64))
    projected = np.column_stack([points @ right, points @ up, points @ camera])
    xy = (projected[:, :2] * 0.82 + 0.5) * (size - 1)
    depth = projected[:, 2]
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = background
    zbuffer = np.full((size, size), -np.inf, dtype=np.float64)
    foreground = np.zeros((size, size), dtype=bool)
    textured = np.zeros((size, size), dtype=bool)
    textures = _load_textures(asset)
    texcoords = np.asarray(asset.texcoords, dtype=np.float64)

    for face_index, face in enumerate(np.asarray(asset.faces, dtype=np.int64)):
        triangle = xy[face]
        minimum = np.maximum(np.floor(triangle.min(axis=0)).astype(int), 0)
        maximum = np.minimum(np.ceil(triangle.max(axis=0)).astype(int), size - 1)
        if np.any(maximum < minimum):
            continue
        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denominator)) < 1e-10:
            continue
        xs, ys = np.meshgrid(
            np.arange(minimum[0], maximum[0] + 1),
            np.arange(minimum[1], maximum[1] + 1),
        )
        w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denominator
        w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        interpolated_depth = w0 * depth[face[0]] + w1 * depth[face[1]] + w2 * depth[face[2]]
        target_depth = zbuffer[ys, xs]
        visible = inside & (interpolated_depth > target_depth)
        if not np.any(visible):
            continue

        material = int(asset.face_materials[face_index])
        texture = textures[material] if 0 <= material < len(textures) else None
        profile = _material_render_profile(asset, material)
        valid_texture = texture is not None and np.isfinite(texcoords[face]).all()
        if valid_texture:
            uv = (
                w0[..., None] * texcoords[face[0]]
                + w1[..., None] * texcoords[face[1]]
                + w2[..., None] * texcoords[face[2]]
            )
            u = _wrap_texture_coordinate(uv[..., 0], profile["wrap_s"])
            v = _wrap_texture_coordinate(uv[..., 1], profile["wrap_t"])
            tx = np.clip(np.rint(u * (texture.shape[1] - 1)).astype(int), 0, texture.shape[1] - 1)
            ty = np.clip(np.rint((1.0 - v) * (texture.shape[0] - 1)).astype(int), 0, texture.shape[0] - 1)
            sampled = texture[ty, tx].astype(np.float64) / 255.0
        else:
            sampled = np.ones((*xs.shape, 4), dtype=np.float64)

        factor = profile["factor"]
        colors = np.clip(sampled[..., :3] * factor[:3], 0.0, 1.0)
        source_alpha = np.clip(sampled[..., 3] * factor[3], 0.0, 1.0)
        alpha_mode = profile["alpha_mode"]
        if alpha_mode == "OPAQUE":
            alpha = np.ones_like(source_alpha)
            accepted = visible
        elif alpha_mode == "MASK":
            accepted = visible & (source_alpha >= profile["alpha_cutoff"])
            alpha = np.ones_like(source_alpha)
        else:  # BLEND and unknown extension modes
            accepted = visible & (source_alpha > 0.0)
            alpha = source_alpha
        if not np.any(accepted):
            continue

        region = image[ys, xs].astype(np.float64) / 255.0
        blended = np.clip(
            colors * alpha[..., None] + region * (1.0 - alpha[..., None]), 0.0, 1.0
        )
        output = image[ys, xs]
        output[accepted] = np.rint(blended[accepted] * 255.0).astype(np.uint8)
        image[ys, xs] = output
        target_depth[accepted] = interpolated_depth[accepted]
        zbuffer[ys, xs] = target_depth
        foreground_region = foreground[ys, xs]
        foreground_region[accepted] = True
        foreground[ys, xs] = foreground_region
        textured_region = textured[ys, xs]
        if valid_texture:
            textured_region[accepted] = True
        else:
            textured_region[accepted] = False
        textured[ys, xs] = textured_region
    return image, foreground, textured


STANDARD_DIRECTIONS = (
    (1.0, 0.0, 0.35), (0.0, 1.0, 0.35), (-1.0, 0.0, 0.35),
    (0.0, -1.0, 0.35), (1.0, 1.0, 0.7), (0.0, 0.0, 1.0),
)


def simulate_camera_capture(image: np.ndarray, severity: float, seed: int) -> np.ndarray:
    """Procedural screenshot/camera-domain proxy; not a photogrammetric reconstruction."""
    import cv2

    severity = float(np.clip(severity, 0.0, 1.0)); rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    source = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    jitter = (0.03 + 0.07 * severity) * min(width, height)
    destination = source + rng.uniform(-jitter, jitter, size=(4, 2)).astype(np.float32)

    # Replace the known renderer background with an independent low-frequency scene.
    y, x = np.mgrid[0:height, 0:width]
    color_a = rng.integers(25, 180, size=3); color_b = rng.integers(25, 180, size=3)
    blend = ((x / max(width - 1, 1)) * 0.6 + (y / max(height - 1, 1)) * 0.4)[..., None]
    background = color_a * (1.0 - blend) + color_b * blend
    foreground = np.any(np.abs(image.astype(np.int16) - 245) > 8, axis=2)
    composite = background.astype(np.uint8); composite[foreground] = image[foreground]
    matrix = cv2.getPerspectiveTransform(source, destination)
    captured = cv2.warpPerspective(composite, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)

    illumination = 0.65 + 0.45 * (x / max(width - 1, 1))
    illumination *= float(rng.uniform(0.75, 1.1))
    captured = np.clip(captured.astype(np.float32) * illumination[..., None], 0, 255)
    captured += rng.normal(0.0, 3.0 + 12.0 * severity, size=captured.shape)
    captured = np.clip(captured, 0, 255).astype(np.uint8)
    if severity >= 0.45:
        kernel = np.zeros((5, 5), dtype=np.float32); kernel[2] = 0.2
        captured = cv2.filter2D(captured, -1, kernel)
    side = int(round(np.sqrt(0.03 + 0.12 * severity) * min(width, height)))
    left = int(rng.integers(0, max(1, width - side + 1))); top = int(rng.integers(0, max(1, height - side + 1)))
    captured[top:top + side, left:left + side] = rng.integers(20, 210, size=3, dtype=np.uint8)
    quality = int(round(82 - 42 * severity))
    encode_ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(captured, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not encode_ok:
        raise RuntimeError("JPEG simulation failed")
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def render_silhouette_view(
    asset: MeshAsset, direction: tuple[float, float, float], size: int = 96
) -> np.ndarray:
    import cv2

    vertices = np.asarray(asset.vertices, dtype=np.float64)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    scale = max(float(np.max(vertices.max(axis=0) - vertices.min(axis=0))), 1e-12)
    points = (vertices - center) / scale
    right, up, _ = _camera_basis(np.asarray(direction, dtype=np.float64))
    xy = (np.column_stack([points @ right, points @ up]) * 0.82 + 0.5) * (size - 1)
    mask = np.zeros((size, size), dtype=np.uint8)
    for face in np.asarray(asset.faces, dtype=np.int64):
        polygon = np.rint(xy[face]).astype(np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def visual_hull_reconstruct(asset: MeshAsset, resolution: int = 32, image_size: int = 96) -> MeshAsset:
    """Controlled orthographic silhouette reconstruction, not photogrammetry."""
    if resolution < 8:
        raise ValueError("visual hull resolution must be at least 8")
    masks = [render_silhouette_view(asset, direction, image_size) for direction in STANDARD_DIRECTIONS]
    axis = np.linspace(-0.5, 0.5, resolution)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    occupied = np.ones(len(grid), dtype=bool)
    for direction, mask in zip(STANDARD_DIRECTIONS, masks):
        right, up, _ = _camera_basis(np.asarray(direction, dtype=np.float64))
        xy = (np.column_stack([grid @ right, grid @ up]) * 0.82 + 0.5) * (image_size - 1)
        pixels = np.rint(xy).astype(np.int64)
        inside = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < image_size)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < image_size)
        )
        visible = np.zeros(len(grid), dtype=bool)
        visible[inside] = mask[pixels[inside, 1], pixels[inside, 0]] > 0
        occupied &= visible
    volume = occupied.reshape(resolution, resolution, resolution)
    if not volume.any() or volume.all():
        raise ValueError("visual hull occupancy is degenerate")
    from skimage.measure import marching_cubes

    vertices, faces, _, _ = marching_cubes(
        volume.astype(np.float32), level=0.5,
        spacing=(1.0 / (resolution - 1),) * 3,
    )
    vertices -= 0.5
    faces = faces.astype(np.int64)
    return MeshAsset(
        vertices=vertices.astype(np.float64), faces=faces,
        normals=recompute_vertex_normals(vertices, faces),
        face_materials=np.full(len(faces), -1, dtype=np.int32),
        metadata={
            "gltf_path": "visual_hull_reconstruction", "resolution": int(resolution),
            "source": asset.metadata.get("asset_id"), "photogrammetric": False,
        },
    )
