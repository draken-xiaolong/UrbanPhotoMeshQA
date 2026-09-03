"""Deterministic multi-view rendering and texture feature extraction."""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .texture import STANDARD_DIRECTIONS, render_textured_view


class ImageEncoder:
    """Frozen MobileNetV3 encoder for six deterministic rendered views."""

    def __init__(self, device):
        weights = MobileNet_V3_Small_Weights.DEFAULT
        network = mobilenet_v3_small(weights=weights).to(device).eval()
        self.features = network.features
        self.pool = network.avgpool
        self.transform = weights.transforms()
        self.device = device

    @torch.no_grad()
    def __call__(self, views):
        batch = torch.stack([self.transform(Image.fromarray(view)) for view in views]).to(self.device)
        embeddings = F.normalize(self.pool(self.features(batch)).flatten(1), dim=1)
        pooled = F.normalize(embeddings.mean(dim=0), dim=0)
        return {
            "pooled": pooled.cpu().numpy().astype(np.float32),
            "views": embeddings.cpu().numpy().astype(np.float32),
        }


def render_views(mesh, size, directions=STANDARD_DIRECTIONS, background=(245, 245, 245)):
    return [
        render_textured_view(mesh, direction=direction, size=size, background=background)
        for direction in directions
    ]


def legacy_image_variant(views, attack, seed=2026):
    """Old rendered-image protocol retained only to reproduce the legacy baseline."""
    output = []
    rng = np.random.default_rng(seed)
    for view in views:
        image = Image.fromarray(view)
        if attack == "jpeg50":
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=50)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        elif attack == "blur1.5":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
        elif attack == "brightness0.55":
            image = ImageEnhance.Brightness(image).enhance(0.55)
        elif attack == "downsample32":
            image = image.resize((32, 32), Image.Resampling.BILINEAR).resize(
                image.size, Image.Resampling.BILINEAR
            )
        elif attack == "occlusion20":
            draw = ImageDraw.Draw(image)
            width, height = image.size
            side = int(round(np.sqrt(0.2) * min(width, height)))
            x = int(rng.integers(0, max(1, width - side + 1)))
            y = int(rng.integers(0, max(1, height - side + 1)))
            draw.rectangle((x, y, x + side, y + side), fill=(128, 128, 128))
        else:
            raise ValueError(attack)
        output.append(np.asarray(image, dtype=np.uint8))
    return output
