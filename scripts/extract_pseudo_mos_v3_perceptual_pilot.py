#!/usr/bin/env python3
"""Extract a six-view, multi-layer perceptual teacher for pseudo-MOS v3 Pilot cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbanphotomeshqa.gltf import GltfReader  # noqa: E402
from urbanphotomeshqa.texture import (  # noqa: E402
    STANDARD_DIRECTIONS,
    render_textured_view_with_masks,
)


LEVELS = ("light", "medium", "heavy")
LAYERS = (1, 3, 8, 12)


def render(path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = GltfReader(path).load_mesh(include_texture=True)
    values = [render_textured_view_with_masks(mesh, direction, size)
              for direction in STANDARD_DIRECTIONS]
    return np.stack([item[0] for item in values]), np.stack([item[1] for item in values])


class PerceptualTeacher:
    def __init__(self, device: torch.device):
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.features = mobilenet_v3_small(weights=weights).features.to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
        self.device = device

    @torch.no_grad()
    def maps(self, images: np.ndarray) -> list[torch.Tensor]:
        x = torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        output = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in LAYERS:
                output.append(F.normalize(x, dim=1))
        return output

    @torch.no_grad()
    def compare(self, clean_rgb: np.ndarray, clean_mask: np.ndarray,
                attacked_rgb: np.ndarray, attacked_mask: np.ndarray) -> dict:
        clean_maps, attacked_maps = self.maps(clean_rgb), self.maps(attacked_rgb)
        union = torch.from_numpy(clean_mask | attacked_mask).to(self.device).float()[:, None]
        intersection = np.logical_and(clean_mask, attacked_mask).sum(axis=(1, 2))
        union_count = np.logical_or(clean_mask, attacked_mask).sum(axis=(1, 2))
        layer_distances = []
        layer_p95 = []
        layer_worst_views = []
        for clean, attacked in zip(clean_maps, attacked_maps):
            weight = F.interpolate(union, size=clean.shape[-2:], mode="area")
            distance = ((clean - attacked) ** 2).mean(dim=1, keepdim=True)
            per_view = (distance * weight).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3)).clamp_min(1e-6)
            layer_distances.append(float(per_view.mean()))
            layer_worst_views.append(float(per_view.max()))
            visible_values = distance[weight > 0.05]
            layer_p95.append(float(torch.quantile(visible_values, 0.95)) if len(visible_values) else 0.0)
        clean_value = clean_rgb.astype(np.float32) / 255.0
        attacked_value = attacked_rgb.astype(np.float32) / 255.0
        pixel_union = clean_mask | attacked_mask
        rgb_rmse = float(np.sqrt(np.mean((clean_value[pixel_union] - attacked_value[pixel_union]) ** 2)))
        return {
            "feature_layer_mse": layer_distances,
            "feature_mean": float(np.mean(layer_distances)),
            "feature_layer_p95": layer_p95,
            "feature_p95_mean": float(np.mean(layer_p95)),
            "feature_worst_view_mean": float(np.mean(layer_worst_views)),
            "rgb_rmse_union": rgb_rmse,
            "silhouette_iou": float(np.mean(intersection / np.maximum(union_count, 1))),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    teacher = PerceptualTeacher(device)
    results = []
    for case in cases:
        # Clean path is canonical and independent of the attacked package structure.
        parts = Path(case["levels"]["light"]["gltf_path"]).parts
        clean_path = args.data_root / "HK3D-Individualised" / parts[1] / "BUILDING" / case["asset_id"] / f"{case['asset_id']}.gltf"
        clean_rgb, clean_mask = render(clean_path, args.size)
        level_results = {}
        for level in LEVELS:
            attacked_path = args.data_root / case["levels"][level]["gltf_path"]
            attacked_rgb, attacked_mask = render(attacked_path, args.size)
            metrics = teacher.compare(clean_rgb, clean_mask, attacked_rgb, attacked_mask)
            level_results[level] = {**case["levels"][level], **metrics}
        results.append({"attack": case["attack"], "asset_id": case["asset_id"], "levels": level_results})
        print(case["attack"], case["asset_id"],
              [level_results[level]["feature_mean"] for level in LEVELS], flush=True)
    payload = {
        "schema_version": 1,
        "teacher": "ImageNet MobileNetV3-Small six-view aligned normalized feature distance",
        "layers": list(LAYERS),
        "render_size": args.size,
        "views": len(STANDARD_DIRECTIONS),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
