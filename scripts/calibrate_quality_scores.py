#!/usr/bin/env python3
"""Fit validation-only affine calibration and report locked Test/Blind QA metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_real_gltf_quality import QualityHead, Store, correlation, forward, rankdata


def affine_fit(prediction, truth):
    design = np.column_stack([prediction, np.ones(len(prediction))])
    slope, intercept = np.linalg.lstsq(design, truth, rcond=None)[0]
    return float(max(slope, 0.0)), float(intercept)


def metrics(prediction, truth):
    return {"mae": float(np.mean(np.abs(prediction - truth))),
            "plcc": correlation(prediction, truth),
            "srcc": correlation(rankdata(prediction), rankdata(truth))}


@torch.no_grad()
def predict(model, data):
    model.eval(); output = forward(model, data)
    return {name: output[name].cpu().numpy() for name in ("overall", "geometry", "texture")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--objective-target-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--require-formal", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    normalization = state.get("statistics", {}).get("normalization", "mean_std")
    store = Store(args.feature_dir, device, args.objective_target_dir,
                  args.dataset_manifest, args.require_formal, normalization)
    checkpoint_provenance = state.get("dataset_provenance")
    if checkpoint_provenance is not None:
        fields = ("ordered_sample_sha256", "counts",
                  "excluded_exact_duplicate_severity_packages", "formal")
        if any(checkpoint_provenance.get(field) != store.dataset_provenance.get(field)
               for field in fields):
            raise ValueError("Checkpoint and calibration dataset provenance do not match")
    model = QualityHead(state["dims"], state["branch_indices"], state["use_patches"]).to(device).eval()
    model.load_state_dict(state["model"])
    predictions = {split: predict(model, store.data[split]) for split in ("val", "test", "blind")}
    target_names = {"overall": "overall", "geometry": "geometry", "texture": "texture_quality"}
    calibration, results = {}, {}
    for name, target_name in target_names.items():
        val_truth = store.data["val"][target_name].cpu().numpy()
        slope, intercept = affine_fit(predictions["val"][name], val_truth)
        calibration[name] = {"slope": slope, "intercept": intercept, "fit_split": "val"}
        results[name] = {}
        for split in ("val", "test", "blind"):
            truth = store.data[split][target_name].cpu().numpy()
            raw = predictions[split][name]
            calibrated = np.clip(slope * raw + intercept, 0.0, 1.0)
            results[name][split] = {"raw": metrics(raw, truth), "calibrated": metrics(calibrated, truth)}
    report = {"schema_version": 1, "checkpoint": str(args.checkpoint),
              "dataset_provenance": store.dataset_provenance,
              "protocol": "affine coefficients fit on Val only; Test and Blind remain locked",
              "calibration": calibration, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
