from types import SimpleNamespace

import numpy as np
from PIL import Image

from urbanphotomeshqa.texture_features import (
    ASSET_STAT_NAMES,
    VIEW_STAT_NAMES,
    texture_asset_statistics,
    texture_quality_statistics,
)


def test_texture_quality_statistics_are_finite_and_detect_detail():
    checker = ((np.indices((32, 32)).sum(axis=0) % 2) * 255).astype(np.uint8)
    detailed = np.repeat(checker[:, :, None], 3, axis=2)
    flat = np.full_like(detailed, 127)
    views = np.stack([detailed, flat])
    masks = np.ones((2, 32, 32), dtype=bool)
    statistics = texture_quality_statistics(views, masks, masks)
    assert statistics.shape == (2, len(VIEW_STAT_NAMES))
    assert np.isfinite(statistics).all()
    gradient = VIEW_STAT_NAMES.index("gradient_mean")
    assert statistics[0, gradient] > statistics[1, gradient]


def test_texture_asset_statistics_include_uv_used_texel_density(tmp_path):
    texture = tmp_path / "texture.png"
    Image.new("RGB", (64, 32), (120, 80, 40)).save(texture)
    asset = SimpleNamespace(
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float64),
        faces=np.asarray([[0, 1, 2]], np.int64),
        face_materials=np.asarray([0], np.int32),
        texcoords=np.asarray([[0, 0], [1, 0], [0, 1]], np.float64),
        metadata={"material_texture_paths": [str(texture)]},
    )
    statistics = texture_asset_statistics(asset, raster_size=64)
    assert statistics.shape == (len(ASSET_STAT_NAMES),)
    assert np.isfinite(statistics).all()
    assert statistics[ASSET_STAT_NAMES.index("uv_atlas_coverage")] > 0.45
    assert statistics[ASSET_STAT_NAMES.index("textured_surface_fraction")] == 1.0
