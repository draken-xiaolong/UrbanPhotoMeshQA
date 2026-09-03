#!/usr/bin/env python3
"""Reproducible per-attack, severity, tile and branch-shift QA diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_real_gltf_quality import BRANCHES, QualityHead, correlation, rankdata


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    if len(truth) < 3 or np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        plcc = srcc = None
    else:
        plcc = correlation(prediction, truth)
        srcc = correlation(rankdata(prediction), rankdata(truth))
    return {"count": int(len(truth)), "mae": float(np.mean(np.abs(prediction - truth))),
            "plcc": plcc, "srcc": srcc}


def grouped(prediction, truth, labels, order=None):
    names = order or sorted(np.unique(labels).astype(str).tolist())
    return {name: metrics(prediction[labels == name], truth[labels == name])
            for name in names if np.any(labels == name)}


def load_npz(path: Path) -> dict:
    with np.load(path) as values:
        return {name: values[name].copy() for name in values.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--objective-target-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = QualityHead(state["dims"], state["branch_indices"], state["use_patches"]).to(device).eval()
    model.load_state_dict(state["model"])
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))["calibration"]["overall"]
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))["records"]
    sheets = {row["asset_id"]: row["sheet"] for row in source}

    split_data = {}
    report = {"schema_version": 1, "checkpoint": str(args.checkpoint),
              "protocol": "diagnosis only; never use Test or Blind for model selection",
              "splits": {}, "branch_clean_centroid_shift_rms_train_z": {}}
    for split in ("train", "val", "test", "blind"):
        features = load_npz(args.feature_dir / f"features_{split}.npz")
        targets = load_npz(args.objective_target_dir / f"objective_targets_{split}.npz")
        identity = (np.array_equal(features["asset_ids"], targets["asset_ids"])
                    and np.array_equal(features["attacks"], targets["attacks"])
                    and np.array_equal(features["levels"], targets["levels"]))
        if not identity:
            raise ValueError(f"Feature/target order mismatch: {split}")
        branches = []
        for branch in BRANCHES:
            stats = state["statistics"]["branches"][branch]
            value = ((features[branch] - np.asarray(stats["mean"])) / np.asarray(stats["std"]))
            branches.append(torch.from_numpy(value.astype(np.float32)).to(device))
        patch_mean = np.asarray(state["statistics"]["patch_mean"])
        patch_std = np.asarray(state["statistics"]["patch_std"])
        patches = ((features["patches"] - patch_mean) / patch_std).astype(np.float32)
        patches[~features["patch_mask"]] = 0.0
        with torch.no_grad():
            raw = model(branches, torch.from_numpy(patches).to(device),
                        torch.from_numpy(features["patch_mask"]).to(device))["overall"].cpu().numpy()
        prediction = np.clip(calibration["slope"] * raw + calibration["intercept"], 0.0, 1.0)
        truth = targets["overall_quality"]
        attacks = features["attacks"].astype(str)
        levels = features["levels"].astype(str)
        split_report = {
            "overall": metrics(prediction, truth),
            "per_attack": grouped(prediction, truth, attacks),
            "per_level_attacked_only": grouped(
                prediction, truth, levels, ["light", "medium", "heavy"]),
        }
        if split == "blind":
            tile = np.asarray([sheets[str(asset)] for asset in features["asset_ids"]])
            split_report["per_tile"] = grouped(prediction, truth, tile)
        report["splits"][split] = split_report
        split_data[split] = features

    train = split_data["train"]
    train_clean = train["attacks"].astype(str) == "clean"
    for branch in BRANCHES:
        center = train[branch][train_clean].mean(axis=0)
        scale = np.maximum(train[branch][train_clean].std(axis=0), 1e-5)
        shifts = {}
        for split, features in split_data.items():
            clean = features["attacks"].astype(str) == "clean"
            delta = (features[branch][clean].mean(axis=0) - center) / scale
            shifts[split] = float(np.sqrt(np.mean(delta * delta)))
        report["branch_clean_centroid_shift_rms_train_z"][branch] = shifts

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
