import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


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
