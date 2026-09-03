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
