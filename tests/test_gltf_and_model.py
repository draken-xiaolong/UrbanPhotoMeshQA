from pathlib import Path
from types import SimpleNamespace

import numpy as np

from urbanphotomeshqa.gltf import GltfReader, sample_surface
from urbanphotomeshqa.local_features import extract_local_features
from urbanphotomeshqa.patches import patch_layout, topological_patch_layout
from urbanphotomeshqa.texture import render_textured_view_with_masks
from urbanphotomeshqa.texture_features import patch_texture_atlases


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


def test_texture_renderer_can_return_visible_face_ids():
    asset = GltfReader(first_gltf()).load_mesh(include_texture=True)
    rgb, foreground, textured, face_ids = render_textured_view_with_masks(
        asset, size=32, return_face_ids=True)
    assert rgb.shape == (32, 32, 3)
    assert foreground.shape == textured.shape == face_ids.shape == (32, 32)
    assert np.all(face_ids[foreground] >= 0)
    assert np.all(face_ids[foreground] < len(asset.faces))


def test_patch_texture_atlases_cover_uv_regions():
    asset = GltfReader(first_gltf()).load_mesh(include_texture=True)
    layout = topological_patch_layout(asset, 16)
    images, masks = patch_texture_atlases(asset, layout["face_patch"], size=32)
    assert images.shape == (16, 32, 32, 3)
    assert masks.shape == (16, 32, 32)
    assert masks.any(axis=(1, 2)).all()


def test_shared_local_feature_extractor_shapes():
    class FakeEncoder:
        def patch_tokens(self, views, masks):
            return {"patch_view_tokens": np.zeros((*masks.shape[:2], 576), np.float16),
                    "patch_view_mask": masks.any(axis=(2, 3))}

        def masked_global_tokens(self, images, masks):
            return {"tokens": np.zeros((len(images), 576), np.float16),
                    "mask": masks.any(axis=(1, 2))}

    asset = GltfReader(first_gltf()).load_mesh(include_texture=True)
    values = extract_local_features(asset, FakeEncoder(), render_size=24)
    assert values["patch_descriptors"].shape == (16, 58)
    assert values["patch_view_tokens"].shape == (16, 6, 576)
    assert values["patch_atlas_tokens"].shape == (16, 576)
    assert values["patch_mask"].all()


def test_patch_layout_preserves_exact_face_membership():
    mesh = GltfReader(first_gltf()).load_mesh(include_texture=True)
    descriptors, patch_mask, face_indices, face_mask = patch_layout(mesh, 16, 32)
    assert descriptors.shape == (16, 58)
    assert face_indices.shape == face_mask.shape == (16, 32)
    assert np.all(face_indices[face_mask] >= 0)
    assert np.all(face_indices[face_mask] < len(mesh.faces))
    assert np.all(face_indices[~face_mask] == -1)
    assert np.array_equal(patch_mask, face_mask.any(axis=1))


def test_topological_patch_layout_is_deterministic_complete_partition():
    mesh = GltfReader(first_gltf()).load_mesh(include_texture=True)
    first = topological_patch_layout(mesh, 16)
    second = topological_patch_layout(mesh, 16)
    assert np.array_equal(first["face_patch"], second["face_patch"])
    assert np.array_equal(np.sort(first["patch_face_indices"]), np.arange(len(mesh.faces)))
    assert len(first["face_patch"]) == len(mesh.faces)
    assert np.all(first["face_patch"] >= 0)
    assert np.isclose(first["patch_area"].sum(), first["patch_area"][first["patch_mask"]].sum())
    assert np.array_equal(first["patch_mask"], first["patch_area"] > 0)
    for patch in np.flatnonzero(first["patch_mask"]):
        start, stop = first["patch_offsets"][patch:patch + 2]
        members = first["patch_face_indices"][start:stop]
        assert len(members) == np.sum(first["face_patch"] == patch)


def test_topological_patch_layout_bridges_disconnected_components():
    vertices = [point for i in range(3)
                for point in ([3 * i, 0, 0], [3 * i + 1, 0, 0], [3 * i, 1, 0])]
    mesh = SimpleNamespace(
        vertices=np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
        faces=np.arange(9, dtype=np.int64).reshape(-1, 3))
    layout = topological_patch_layout(mesh, patch_count=2)
    assert int(layout["connected_components"]) == 3
    assert int(layout["virtual_bridge_count"]) == 2
    assert int(layout["patch_mask"].sum()) == 2
    assert sorted(np.unique(layout["face_patch"]).tolist()) == [0, 1]


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
