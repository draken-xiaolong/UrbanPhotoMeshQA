from pathlib import Path

import numpy as np
from PIL import Image

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.gltf_export import export_textured_gltf
from urbanphotomeshqa.integrity import asset_digest
from urbanphotomeshqa.real_attacks import (
    geometry_hole,
    geometry_noise_spike,
    qem_simplify_textured,
    texture_detail_loss,
    texture_misalignment,
    texture_region_missing,
)


FIXTURE = Path(__file__).parent / "fixtures" / "B360011502301063A0" / "B360011502301063A0.gltf"


def test_geometry_attacks_remain_exportable(tmp_path):
    asset = GltfReader(FIXTURE).load_mesh(include_texture=True)
    attacks = [
        geometry_hole(asset, 0.05, 2026),
        geometry_noise_spike(asset, 0.003, 0.07, 2026),
        qem_simplify_textured(asset, 0.7),
    ]
    source_texture = Path(asset.metadata["material_texture_paths"][0])
    for index, attacked in enumerate(attacks):
        output = tmp_path / str(index) / "sample.gltf"
        texture_dir = output.parent / "textures"
        texture_dir.mkdir(parents=True)
        target_texture = texture_dir / source_texture.name
        target_texture.write_bytes(source_texture.read_bytes())
        export_textured_gltf(attacked, output, [source_texture.name])
        digest, dependencies = asset_digest(output)
        reloaded = GltfReader(output).load_mesh(include_texture=True)
        assert digest and len(dependencies) == 3
        assert len(reloaded.faces) > 0
        assert np.isfinite(reloaded.texcoords).all()


def test_source_texture_attacks_have_expected_effects():
    image = Image.open(FIXTURE.parent / "B360011502301063A0_001.jpg")
    blur = texture_detail_loss(image, "gaussian_blur", 2.0)
    lowres = texture_detail_loss(image, "texture_downsample", 0.25)
    missing, actual = texture_region_missing(image, 0.15, 2026)
    shifted = texture_misalignment(image, 0.03, 0.30, 2026)
    assert blur.size == image.size
    assert lowres.width < image.width and lowres.height < image.height
    assert 0.13 <= actual <= 0.17
    assert not np.array_equal(np.asarray(missing.convert("RGB")), np.asarray(image.convert("RGB")))
    assert not np.array_equal(np.asarray(shifted.convert("RGB")), np.asarray(image.convert("RGB")))
