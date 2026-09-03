from pathlib import Path

import numpy as np

from urbanphotomeshqa.gltf import GltfReader, sample_surface
from urbanphotomeshqa.patches import patch_layout


FIXTURE = Path(__file__).parent / "fixtures" / "B360011502301063A0" / "B360011502301063A0.gltf"


def first_gltf() -> Path:
    return FIXTURE


def test_load_and_sample_real_gltf():
    asset = GltfReader(first_gltf()).load_mesh()
    points, stats = sample_surface(asset, 256, 7)
    assert points.shape == (256, 6)
    assert np.isfinite(points).all()
    assert np.max(np.abs(points[:, :3])) <= 1.0
    assert stats["face_count"] > 0


def test_load_textured_real_gltf():
    asset = GltfReader(first_gltf()).load_mesh(include_texture=True)
    assert asset.texcoords is not None
    assert asset.texcoords.shape == (len(asset.vertices), 2)
    assert np.isfinite(asset.texcoords).all()
    paths = asset.metadata["material_texture_paths"]
    assert paths and all(Path(path).exists() for path in paths if path is not None)


def test_patch_layout_preserves_exact_face_membership():
    mesh = GltfReader(first_gltf()).load_mesh(include_texture=True)
    descriptors, patch_mask, face_indices, face_mask = patch_layout(mesh, 16, 32)
    assert descriptors.shape == (16, 58)
    assert face_indices.shape == face_mask.shape == (16, 32)
    assert np.all(face_indices[face_mask] >= 0)
    assert np.all(face_indices[face_mask] < len(mesh.faces))
    assert np.all(face_indices[~face_mask] == -1)
    assert np.array_equal(patch_mask, face_mask.any(axis=1))


def test_model_forward_if_torch_available():
    try:
        import torch
    except ImportError:
        return
    from urbanphotomeshqa.model import BuildingBaseEncoder

    model = BuildingBaseEncoder(input_dim=6, embedding_dim=32, local_dim=32, k=8)
    outputs = model(torch.randn(2, 64, 6))
    assert outputs["identity"].shape == (2, 32)
    assert outputs["local"].shape == (2, 64, 32)
    assert outputs["quality"].shape == (2,)


def test_morphology_targets_are_rigid_and_scale_invariant():
    import math
    import torch

    from urbanphotomeshqa.morphology import global_morphology_targets, local_morphology_targets

    generator = torch.Generator().manual_seed(7)
    points = torch.randn(2, 48, 6, generator=generator)
    points[..., 3:] = torch.nn.functional.normalize(points[..., 3:], dim=-1)
    angle = 0.73
    rotation = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    transformed = points.clone()
    transformed[..., :3] = points[..., :3] @ rotation.T * 1.7 + 3.0
    transformed[..., 3:] = points[..., 3:] @ rotation.T

    assert torch.allclose(
        global_morphology_targets(points), global_morphology_targets(transformed), atol=1e-5
    )
    original_local = local_morphology_targets(points, k=8)
    transformed_local = local_morphology_targets(transformed, k=8)
    # Covariance spectra and normal agreement are scale invariant; the two
    # explicit radial dimensions intentionally retain local relative scale.
    invariant_dimensions = [0, 1, 2, 5, 6]
    assert torch.allclose(
        original_local[..., invariant_dimensions],
        transformed_local[..., invariant_dimensions],
        atol=1e-5,
    )


def test_invariant_encoder_is_so3_invariant_in_eval_mode():
    import torch
    import torch.nn.functional as F

    from urbanphotomeshqa.model import BuildingInvariantEncoder

    generator = torch.Generator().manual_seed(11)
    points = torch.randn(2, 64, 6, generator=generator)
    points[..., 3:] = F.normalize(points[..., 3:], dim=-1)
    quaternion = F.normalize(torch.randn(4, generator=generator), dim=0)
    w, x, y, z = quaternion
    rotation = torch.tensor([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    transformed = points.clone()
    transformed[..., :3] = points[..., :3] @ rotation.T * 1.7 + 2.0
    transformed[..., 3:] = points[..., 3:] @ rotation.T
    model = BuildingInvariantEncoder(embedding_dim=32, local_dim=32, k=8).eval()
    with torch.no_grad():
        original = model(points)["identity"]
        changed = model(transformed)["identity"]
    assert torch.allclose(original, changed, atol=2e-5)
