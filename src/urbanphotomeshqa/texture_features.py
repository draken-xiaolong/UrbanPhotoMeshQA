"""No-reference spatial and statistical texture features for rendered glTF meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


VIEW_STAT_NAMES = (
    "foreground_fraction", "textured_foreground_fraction", "luminance_mean",
    "luminance_std", "gradient_mean", "gradient_p90", "laplacian_abs_mean",
    "entropy_32bin", "saturation_mean", "flat_pixel_fraction", "dark_fraction",
    "bright_fraction",
)
ASSET_STAT_NAMES = (
    "log_total_texture_pixels", "log_uv_used_texture_pixels", "log_surface_area",
    "log_used_texels_per_area", "uv_atlas_coverage", "textured_surface_fraction",
    "log_texture_count", "log_max_texture_dimension",
)


def _entropy(values: np.ndarray) -> float:
    histogram = np.histogram(values, bins=32, range=(0.0, 1.0))[0].astype(np.float64)
    probabilities = histogram / max(histogram.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def texture_quality_statistics(
    views: np.ndarray, foreground_masks: np.ndarray, textured_masks: np.ndarray,
) -> np.ndarray:
    """Return 12 explicit no-reference measurements for each rendered view."""
    output = []
    for image, foreground, textured in zip(views, foreground_masks, textured_masks):
        rgb = np.asarray(image, dtype=np.float64) / 255.0
        foreground = np.asarray(foreground, dtype=bool)
        valid = foreground & np.asarray(textured, dtype=bool)
        if not valid.any():
            valid = foreground
        if not valid.any():
            valid = np.ones(foreground.shape, dtype=bool)
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        dx = np.zeros_like(gray); dy = np.zeros_like(gray)
        dx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
        dy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
        gradient = np.sqrt(dx * dx + dy * dy)
        laplacian = np.abs(-4.0 * gray + np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
                           + np.roll(gray, 1, 1) + np.roll(gray, -1, 1))
        saturation = rgb.max(axis=2) - rgb.min(axis=2)
        selected = gray[valid]; selected_gradient = gradient[valid]
        output.append([
            float(foreground.mean()),
            float((foreground & textured).sum() / max(foreground.sum(), 1)),
            float(selected.mean()), float(selected.std()),
            float(selected_gradient.mean()), float(np.quantile(selected_gradient, 0.90)),
            float(laplacian[valid].mean()), _entropy(selected), float(saturation[valid].mean()),
            float(np.mean(selected_gradient < 0.01)), float(np.mean(selected < 0.08)),
            float(np.mean(selected > 0.92)),
        ])
    result = np.asarray(output, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite rendered texture statistics")
    return result


def texture_asset_statistics(asset, raster_size: int = 512) -> np.ndarray:
    """Summarize texture resolution, UV use, and surface-area-normalized density."""
    paths = asset.metadata.get("material_texture_paths", [])
    canvases: dict[str, Image.Image] = {}
    dimensions: dict[str, tuple[int, int]] = {}
    for value in paths:
        if value and Path(value).is_file() and value not in canvases:
            with Image.open(value) as image:
                dimensions[value] = image.size
            canvases[value] = Image.new("1", (raster_size, raster_size), 0)
    draws = {path: ImageDraw.Draw(canvas) for path, canvas in canvases.items()}
    texcoords = (np.full((len(asset.vertices), 2), np.nan, dtype=np.float64)
                 if asset.texcoords is None else np.asarray(asset.texcoords, dtype=np.float64))
    triangles = np.asarray(asset.vertices, dtype=np.float64)[np.asarray(asset.faces, dtype=np.int64)]
    double_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1)
    total_area = max(float(double_area.sum() * 0.5), 1e-12)
    textured_area = 0.0
    for face_index, face in enumerate(np.asarray(asset.faces, dtype=np.int64)):
        material = int(asset.face_materials[face_index])
        path = paths[material] if 0 <= material < len(paths) else None
        if path not in draws or not np.isfinite(texcoords[face]).all():
            continue
        textured_area += float(double_area[face_index] * 0.5)
        uv = np.clip(texcoords[face], 0.0, 1.0)
        draws[path].polygon([(float(point[0] * (raster_size - 1)),
                              float((1.0 - point[1]) * (raster_size - 1)))
                             for point in uv], fill=1)
    total_pixels = float(sum(width * height for width, height in dimensions.values()))
    used_pixels = 0.0
    for path, canvas in canvases.items():
        coverage = float(np.asarray(canvas, dtype=bool).mean())
        width, height = dimensions[path]
        used_pixels += coverage * width * height
    maximum_dimension = max((max(size) for size in dimensions.values()), default=0)
    result = np.asarray([
        np.log1p(total_pixels), np.log1p(used_pixels), np.log1p(total_area),
        np.log1p(used_pixels / total_area), used_pixels / max(total_pixels, 1.0),
        textured_area / total_area, np.log1p(len(dimensions)), np.log1p(maximum_dimension),
    ], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite asset texture statistics")
    return result


def patch_texture_atlases(asset, face_patch: np.ndarray, size: int = 224):
    """Rasterize each Patch's own UV-used texels into a deterministic material grid."""
    from .texture import _load_textures, _material_render_profile, _wrap_texture_coordinate

    textures = _load_textures(asset)
    usable = [index for index, texture in enumerate(textures) if texture is not None]
    columns = max(1, int(np.ceil(np.sqrt(len(usable))))); rows = max(1, int(np.ceil(len(usable) / columns)))
    material_cell = {material: (position // columns, position % columns)
                     for position, material in enumerate(usable)}
    texcoords = np.asarray(asset.texcoords, np.float64)
    images = np.full((int(face_patch.max()) + 1, size, size, 3), 245, np.uint8)
    masks = np.zeros(images.shape[:3], bool)
    for patch in range(len(images)):
        for material in usable:
            face_ids = np.flatnonzero((face_patch == patch) &
                                      (np.asarray(asset.face_materials) == material))
            if not len(face_ids):
                continue
            row, column = material_cell[material]
            x0, x1 = column * size // columns, (column + 1) * size // columns
            y0, y1 = row * size // rows, (row + 1) * size // rows
            texture = textures[material]
            profile = _material_render_profile(asset, material)
            resized = np.asarray(Image.fromarray(texture).resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR))
            factor = profile["factor"]
            rgb = np.clip(resized[..., :3].astype(np.float64) * factor[:3], 0, 255).astype(np.uint8)
            alpha = resized[..., 3].astype(np.float64) / 255.0 * factor[3]
            material_mask = Image.new("1", (size, size), 0); draw = ImageDraw.Draw(material_mask)
            for face_id in face_ids:
                uv = texcoords[np.asarray(asset.faces[face_id], np.int64)]
                if not np.isfinite(uv).all():
                    continue
                u = _wrap_texture_coordinate(uv[:, 0], profile["wrap_s"])
                v = _wrap_texture_coordinate(uv[:, 1], profile["wrap_t"])
                draw.polygon([(x0 + float(value_u) * max(x1-x0-1, 1),
                               y0 + (1-float(value_v)) * max(y1-y0-1, 1))
                              for value_u, value_v in zip(u, v)], fill=1)
            current = np.asarray(material_mask, bool)
            if profile["alpha_mode"] == "MASK":
                alpha_valid = alpha >= profile["alpha_cutoff"]
            elif profile["alpha_mode"] == "OPAQUE":
                alpha_valid = np.ones_like(alpha, dtype=bool)
            else:
                alpha_valid = alpha > 0.0
            cell_mask = current[y0:y1, x0:x1] & alpha_valid
            images[patch, y0:y1, x0:x1][cell_mask] = rgb[cell_mask]
            masks[patch, y0:y1, x0:x1] |= cell_mask
    return images, masks


class SpatialImageEncoder:
    """Frozen MobileNet feature maps retained as foreground-aware multi-view tokens."""

    def __init__(self, device):
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.features = mobilenet_v3_small(weights=weights).features.to(device).eval()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    @torch.no_grad()
    def __call__(self, views: np.ndarray, textured_masks: np.ndarray) -> dict[str, np.ndarray]:
        images = torch.from_numpy(np.asarray(views)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        images = (images - self.mean) / self.std
        feature_map = self.features(images)
        masks = torch.from_numpy(np.asarray(textured_masks)).to(self.device).float()[:, None]
        masks = F.interpolate(masks, size=feature_map.shape[-2:], mode="area")
        tokens, valid = [], []
        global_denominator = masks.sum(dim=(2, 3)).clamp_min(1e-6)
        tokens.append((feature_map * masks).sum(dim=(2, 3)) / global_denominator)
        valid.append((global_denominator[:, 0] > 0.05))
        height, width = feature_map.shape[-2:]
        for row in range(2):
            for column in range(2):
                y0, y1 = row * height // 2, (row + 1) * height // 2
                x0, x1 = column * width // 2, (column + 1) * width // 2
                local_mask = masks[:, :, y0:y1, x0:x1]
                denominator = local_mask.sum(dim=(2, 3)).clamp_min(1e-6)
                local = feature_map[:, :, y0:y1, x0:x1]
                tokens.append((local * local_mask).sum(dim=(2, 3)) / denominator)
                valid.append((denominator[:, 0] > 0.05))
        token_tensor = torch.stack(tokens, dim=1)
        valid_tensor = torch.stack(valid, dim=1)
        return {"tokens": token_tensor.cpu().numpy().astype(np.float16),
                "token_mask": valid_tensor.cpu().numpy()}

    @torch.no_grad()
    def patch_tokens(self, views: np.ndarray, patch_masks: np.ndarray) -> dict[str, np.ndarray]:
        """Pool one frozen feature token per Patch and camera view."""
        images = torch.from_numpy(np.asarray(views)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        feature_map = self.features((images - self.mean) / self.std)
        masks = torch.from_numpy(np.asarray(patch_masks)).to(self.device).float()
        patch_count, view_count = masks.shape[:2]
        resized = F.interpolate(masks.reshape(-1, 1, *masks.shape[-2:]),
                                size=feature_map.shape[-2:], mode="area")
        resized = resized.reshape(patch_count, view_count, 1, *feature_map.shape[-2:])
        expanded = feature_map[None].expand(patch_count, -1, -1, -1, -1)
        denominator = resized.sum(dim=(-2, -1)).clamp_min(1e-6)
        pooled = (expanded * resized).sum(dim=(-2, -1)) / denominator
        valid = resized.sum(dim=(2, 3, 4)) > 0.05
        return {"patch_view_tokens": pooled.cpu().numpy().astype(np.float16),
                "patch_view_mask": valid.cpu().numpy()}

    @torch.no_grad()
    def masked_global_tokens(self, images: np.ndarray, masks: np.ndarray) -> dict[str, np.ndarray]:
        values = torch.from_numpy(np.asarray(images)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        feature_map = self.features((values - self.mean) / self.std)
        weights = torch.from_numpy(np.asarray(masks)).to(self.device).float()[:, None]
        weights = F.interpolate(weights, size=feature_map.shape[-2:], mode="area")
        denominator = weights.sum(dim=(2, 3)).clamp_min(1e-6)
        tokens = (feature_map * weights).sum(dim=(2, 3)) / denominator
        return {"tokens": tokens.cpu().numpy().astype(np.float16),
                "mask": np.asarray(masks, dtype=bool).any(axis=(1, 2))}
