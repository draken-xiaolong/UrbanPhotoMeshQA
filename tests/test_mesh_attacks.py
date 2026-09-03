from pathlib import Path

import numpy as np
import torch

from urbanphotomeshqa.gltf import GltfReader, sample_surface
from urbanphotomeshqa.mesh_attacks import apply_mesh_attack, mesh_face_graph, topology_stats
from urbanphotomeshqa.model import MeshFaceEncoder
from urbanphotomeshqa.texture import apply_uv_preserving_attack, render_textured_view, simulate_camera_capture


FIXTURE = Path(__file__).parent / "fixtures" / "B360011502301063A0" / "B360011502301063A0.gltf"


def _sample_asset(include_texture: bool = False):
    return GltfReader(FIXTURE).load_mesh(include_texture=include_texture)


def test_real_mesh_attacks_and_sampling():
    asset = _sample_asset()
    attacks = [
        ("qem", 0.5), ("connected_crop", 0.3), ("hole", 0.15),
        ("normal_flip", 0.25), ("retriangulate", 0.5),
    ]
    for attack, severity in attacks:
        points, _ = sample_surface(apply_mesh_attack(asset, attack, severity, 7), 128, 8)
        assert points.shape == (128, 6)
        assert np.isfinite(points).all()


def test_retriangulation_preserves_area_and_changes_faces():
    asset = _sample_asset()
    before = topology_stats(asset)
    after = topology_stats(apply_mesh_attack(asset, "retriangulate", 1.0, 7))
    assert after["face_count"] > before["face_count"]
    assert np.isclose(after["surface_area"], before["surface_area"], rtol=1e-8)


def test_native_face_graph_forward():
    features, neighbors, topology = mesh_face_graph(_sample_asset())
    output = MeshFaceEncoder()(
        torch.from_numpy(features)[None],
        torch.from_numpy(neighbors)[None],
        torch.ones(1, len(features), dtype=torch.bool),
        torch.from_numpy(topology)[None],
    )
    assert output["identity"].shape == (1, 256)
    assert output["local"].shape[:2] == (1, len(features))


def test_uv_attacks_and_cpu_rendering():
    asset = _sample_asset(include_texture=True)
    assert asset.texcoords is not None
    for attack, severity in [("connected_crop", 0.2), ("hole", 0.1), ("normal_flip", 0.2), ("retriangulate", 0.2)]:
        attacked = apply_uv_preserving_attack(asset, attack, severity, 7)
        assert attacked.texcoords is not None and len(attacked.texcoords) == len(attacked.vertices)
        rendered = render_textured_view(attacked, size=64)
        assert rendered.shape == (64, 64, 3)
        assert rendered.dtype == np.uint8


def test_simulated_capture_is_deterministic_and_nontrivial():
    image = np.full((64, 64, 3), 245, dtype=np.uint8)
    image[16:48, 20:44] = np.asarray([80, 120, 160], dtype=np.uint8)
    first = simulate_camera_capture(image, 0.5, 7)
    second = simulate_camera_capture(image, 0.5, 7)
    assert np.array_equal(first, second)
    assert first.shape == image.shape and first.dtype == np.uint8
    assert not np.array_equal(first, image)
