#!/usr/bin/env python3
"""Compare the original and modality-aware quality students when texture is absent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_branch_aware_multitask_quality import build_samples, stats
from train_four_branch_fusion import FeatureStore
from train_modality_aware_quality_student import evaluate_missing_texture
from train_no_reference_quality_student import NoReferenceQualityStudent, attach_patch_features, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--geometry-target-dir", type=Path, required=True)
    parser.add_argument("--texture-target-dir", type=Path, required=True)
    parser.add_argument("--patch-feature-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--modality-aware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    store = FeatureStore(args.feature_dir, device)
    samples, geometry_names, texture_names = build_samples(
        store, args.feature_dir, args.geometry_target_dir, args.texture_target_dir, device)
    patch_dim = attach_patch_features(samples, args.patch_feature_dir, device)
    geometry_stats = stats(samples, "geometry", "geometry_mask")
    texture_stats = stats(samples, "texture", "texture_mask")
    severity_mean = samples["train"]["severity"].mean()
    severity_baseline = float(torch.mean(torch.abs(samples["val"]["severity"] - severity_mean)))
    output = {"protocol": "clean and geometry-degraded samples with texture unavailable"}
    for name, checkpoint_path in (("original", args.baseline), ("modality_aware", args.modality_aware)):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = NoReferenceQualityStudent(
            store.dims, len(geometry_names), len(texture_names), patch_dim=patch_dim,
            modality_aware=bool(checkpoint.get("modality_aware", False))).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        output[name] = {}
        for split in ("test", "blind"):
            output[name][split] = {
                "normal": evaluate(model, samples[split], geometry_stats, texture_stats,
                                   geometry_names, texture_names, severity_baseline),
                "missing_texture": evaluate_missing_texture(
                    model, samples[split], geometry_stats, geometry_names, severity_baseline),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {model: {split: values["missing_texture"] for split, values in splits.items()}
               for model, splits in output.items() if model != "protocol"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
