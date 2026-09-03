#!/usr/bin/env python3
"""Export per-model OQI reports and locked Test/Blind monotonicity checks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from train_branch_aware_multitask_quality import build_samples, stats
from train_four_branch_fusion import FeatureStore
from train_frozen_base_quality_head import ATTACKS, seed_all
from train_no_reference_quality_student import NoReferenceQualityStudent, attach_patch_features
from train_objective_quality_index import (
    OrdinalWeightedPatchHead, encode_student, metric_scales, objective_quality_targets,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--geometry-target-dir", type=Path, required=True)
    parser.add_argument("--texture-target-dir", type=Path, required=True)
    parser.add_argument("--patch-feature-dir", type=Path, required=True)
    parser.add_argument("--quality-student", type=Path, required=True)
    parser.add_argument("--quality-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def metadata(feature_dir, split):
    with np.load(feature_dir / f"scores_{split}.npz") as values:
        query_ids = values["query_ids"].astype(str)
        attacks = values["attacks"].astype(str)
        raw_severity = values["severities"].astype(np.float32)
        targets = values["targets"].astype(np.int64)
        gallery_count = len(values["gallery_point"])
    gallery_ids = np.asarray([query_ids[np.flatnonzero(targets == index)[0]]
                              for index in range(gallery_count)])
    return {
        "asset_ids": np.concatenate([gallery_ids, query_ids]),
        "attacks": np.concatenate([np.full(gallery_count, "clean"), attacks]),
        "raw_severity": np.concatenate([np.zeros(gallery_count, np.float32), raw_severity]),
        "gallery_count": gallery_count,
    }


def monotonicity(rows):
    by_asset_attack = defaultdict(list)
    clean_score = {}
    for row in rows:
        if row["attack"] == "clean":
            clean_score[row["asset_id"]] = row["oqi"]
        else:
            by_asset_attack[(row["asset_id"], row["attack"])].append(row)
    comparisons, correct, inversions = 0, 0, []
    for group in by_asset_attack.values():
        levels = sorted(group, key=lambda x: x["severity"])
        for low, high in zip(levels, levels[1:]):
            if high["severity"] <= low["severity"]:
                continue
            comparisons += 1
            correct += int(high["oqi"] <= low["oqi"])
            if high["oqi"] > low["oqi"]:
                inversions.append(high["oqi"] - low["oqi"])
    clean_pairs = [(clean_score[row["asset_id"]], row["oqi"]) for row in rows
                   if row["attack"] != "clean" and row["asset_id"] in clean_score]
    by_attack = {}
    attacks = sorted({row["attack"] for row in rows if row["attack"] != "clean"})
    for attack in attacks:
        values = [row for row in rows if row["attack"] == attack]
        by_level = defaultdict(list)
        for row in values:
            by_level[float(row["severity"])].append(row["oqi"])
        by_attack[attack] = {str(level): float(np.mean(scores)) for level, scores in sorted(by_level.items())}
    return {
        "multi_level_comparisons": comparisons,
        "severity_monotonic_accuracy": correct / max(comparisons, 1),
        "mean_inversion_points": float(np.mean(inversions)) if inversions else 0.0,
        "clean_above_attacked_rate": float(np.mean([clean > attacked for clean, attacked in clean_pairs])),
        "clean_attack_pairs": len(clean_pairs),
        "mean_oqi_by_attack_and_severity": by_attack,
    }


@torch.no_grad()
def main():
    args = parse_args(); seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required for formal report export")
    store = FeatureStore(args.feature_dir, device)
    samples, geometry_names, texture_names = build_samples(
        store, args.feature_dir, args.geometry_target_dir, args.texture_target_dir, device)
    patch_input_dim = attach_patch_features(samples, args.patch_feature_dir, device)
    geometry_stats = stats(samples, "geometry", "geometry_mask")
    texture_stats = stats(samples, "texture", "texture_mask")
    scales = metric_scales(samples); targets = objective_quality_targets(samples, scales)
    student_state = torch.load(args.quality_student, map_location=device, weights_only=True)
    geometry_only = "geometry_only" in student_state.get("selected_variant", "")
    student = NoReferenceQualityStudent(
        store.dims, len(geometry_names), len(texture_names), patch_dim=patch_input_dim,
        patch_geometry_only=geometry_only).to(device).eval()
    student.load_state_dict(student_state["model"])
    encoded = encode_student(student, samples, geometry_stats, texture_stats, scales)
    index_state = torch.load(args.quality_index, map_location=device, weights_only=True)
    ordinal = OrdinalWeightedPatchHead(
        encoded["train"]["global"].shape[1], encoded["train"]["patch"].shape[2]).to(device).eval()
    ordinal.load_state_dict(index_state["model"])
    ordinal_weight = float(index_state.get("ordinal_weight", 0.75))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "ordinal_weight": ordinal_weight, "splits": {}}
    for split in ("test", "blind"):
        data = encoded[split]
        ordinal_score, aux = ordinal(data["global"], data["patch"], data["patch_mask"])
        oqi = 100.0 * (ordinal_weight * ordinal_score + (1.0 - ordinal_weight) * data["deterministic"])
        local_scores = 100.0 * torch.sigmoid(ordinal.patch_head(data["patch"])).mean(2)
        risk = (1.0 - local_scores / 100.0) * aux["patch_weights"]
        meta = metadata(args.feature_dir, split)
        with np.load(args.patch_feature_dir / f"patches_{split}.npz") as values:
            raw_patch = np.concatenate([values["gallery_patch"], values["query_patch"]], 0)
        rows = []
        probability = torch.softmax(student(
            samples[split]["degraded"], samples[split]["patch"], samples[split]["patch_mask"])["attack"], 1)
        predicted = probability.argmax(1)
        for index in range(len(oqi)):
            valid = torch.nonzero(data["patch_mask"][index], as_tuple=False).flatten().cpu().numpy()
            worst = valid[np.argsort(-risk[index, valid].cpu().numpy())[: min(3, len(valid))]]
            rows.append({
                "asset_id": str(meta["asset_ids"][index]), "attack": str(meta["attacks"][index]),
                "severity": float(samples[split]["severity"][index]),
                "oqi": float(oqi[index]), "target_oqi": float(100.0 * targets[split][index]),
                "predicted_attack": ATTACKS[int(predicted[index])],
                "confidence": float(probability[index, predicted[index]]),
                "low_quality_patches": ";".join(str(int(value)) for value in worst),
                "low_patch_centers": json.dumps([raw_patch[index, value, :3].round(5).tolist() for value in worst]),
            })
        csv_path = args.output_dir / f"quality_reports_{split}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        monotonic = monotonicity(rows)
        summary["splits"][split] = {"count": len(rows), "monotonicity": monotonic,
                                    "report_csv": str(csv_path.resolve())}
    (args.output_dir / "monotonicity.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
