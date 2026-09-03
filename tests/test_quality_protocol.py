import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rankdata_uses_average_ranks_for_ties():
    module = load_script("train_real_gltf_quality")
    actual = module.rankdata(np.asarray([30.0, 10.0, 20.0, 20.0]))
    assert np.array_equal(actual, np.asarray([3.0, 0.0, 1.5, 1.5]))


def test_regression_metrics_returns_exact_perfect_spearman_with_ties():
    module = load_script("train_real_gltf_quality")
    values = np.asarray([0.1, 0.2, 0.2, 0.7], dtype=np.float32)
    metrics = module.regression_metrics(values, values)
    assert metrics["count"] == 4
    assert metrics["mae"] == 0.0
    assert np.isclose(metrics["srcc"], 1.0)


def test_v2_objective_targets_align_to_filtered_feature_order(tmp_path):
    module = load_script("train_real_gltf_quality")
    features = {
        "asset_ids": np.asarray(["b", "a"]),
        "attacks": np.asarray(["geometry_hole", "clean"]),
        "levels": np.asarray(["light", "clean"]),
    }
    objective = {
        "asset_ids": np.asarray(["a", "b", "c"]),
        "attacks": np.asarray(["clean", "geometry_hole", "texture_detail_loss"]),
        "levels": np.asarray(["clean", "light", "heavy"]),
    }
    assert np.array_equal(module.align_objective_targets(features, objective, "train"), [1, 0])
    np.savez(tmp_path / "objective_targets_v2_train.npz", **objective)
    assert module.objective_target_path(tmp_path, "train").name == "objective_targets_v2_train.npz"


def test_store_selects_global_features_and_accepts_targets_without_patch_quality(tmp_path):
    module = load_script("train_real_gltf_quality")
    feature_dir, target_dir = tmp_path / "features", tmp_path / "targets"
    feature_dir.mkdir(); target_dir.mkdir()
    for split in ("train", "val"):
        common = {
            "asset_ids": np.asarray(["a"]), "attacks": np.asarray(["clean"]),
            "levels": np.asarray(["clean"]), "attack_index": np.asarray([0]),
            "severity": np.asarray([0], np.float32),
            "overall_quality": np.asarray([0], np.float32),
            "geometry_quality": np.asarray([0], np.float32),
            "texture_quality": np.asarray([0], np.float32),
        }
        np.savez(feature_dir / f"features_{split}.npz", **common,
                 point_identity=np.ones((1, 2), np.float32), point_global=np.ones((1, 3), np.float32),
                 mesh_identity=np.ones((1, 4), np.float32), mesh_global=np.ones((1, 5), np.float32),
                 morphology=np.ones((1, 2), np.float32), texture=np.ones((1, 1), np.float32),
                 patches=np.ones((1, 1, 58), np.float32), patch_mask=np.ones((1, 1), bool))
        np.savez(target_dir / f"objective_targets_v2_{split}.npz",
                 asset_ids=common["asset_ids"], attacks=common["attacks"], levels=common["levels"],
                 overall_quality=np.asarray([1], np.float32),
                 geometry_quality=np.asarray([1], np.float32),
                 texture_quality=np.asarray([1], np.float32))
    store = module.Store(feature_dir, torch.device("cpu"), target_dir,
                         splits=("train", "val"), base_representation="global")
    assert store.dims[:2] == [3, 5]
    assert store.has_patch_quality is False
    assert float(store.data["train"]["overall"][0]) == 1.0


def test_val_summary_applies_gate_without_locked_metrics(tmp_path):
    config = json.loads((ROOT / "configs/quality_generalization_minimal_seed2026.json").read_text())
    config.pop("release_val_reference", None)
    config["runs"] = config["runs"][:4]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    candidates = {
        "B0_formal_frozen": (0.50, 0.60, 0.60),
        "B1_robust_norm": (0.52, 0.59, 0.59),
        "B2_robust_tile_balanced": (0.53, 0.55, 0.61),
        "B3_robust_tile_worst": (0.505, 0.61, 0.61),
    }
    for name, (overall, geometry, texture) in candidates.items():
        directory = tmp_path / name
        directory.mkdir()
        payload = {
            "status": "COMPLETE",
            "seed": 2026,
            "protocol": {"dataset_provenance": {
                "formal": True, "ordered_sample_sha256": "formal-sha"
            }},
            "variants": {"four_branch": {"best_epoch": 7, "results": {"val": {
                "overall": {"srcc": overall, "plcc": overall, "mae": 0.2},
                "geometry": {"srcc": geometry}, "texture": {"srcc": texture},
                "per_tile": {},
            }}}},
        }
        (directory / "results.json").write_text(json.dumps(payload), encoding="utf-8")
        (directory / "four_branch.pt").write_bytes(b"checkpoint")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/summarize_quality_generalization_val.py"),
        "--config", str(config_path), "--run-root", str(tmp_path),
    ], check=True, capture_output=True, text=True)
    selection = json.loads((tmp_path / "val_selection.json").read_text(encoding="utf-8"))
    assert selection["selected_id"] == "B1_robust_norm"
    assert selection["protocol"]["test_blind_evaluated"] is False
