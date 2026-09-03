#!/usr/bin/env python3
"""Calibrate a single-mesh 0-100 objective quality index with ranking-aware heads."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_branch_aware_multitask_quality import build_samples, correlation, rankdata, stats
from train_four_branch_fusion import FeatureStore
from train_frozen_base_quality_head import ATTACKS, seed_all
from train_no_reference_quality_student import (
    NoReferenceQualityStudent,
    attach_patch_features,
    model_forward,
)


GEOMETRY_CLASSES = (5, 7, 10)  # connected_crop, hole, retriangulate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--geometry-target-dir", type=Path, required=True)
    parser.add_argument("--texture-target-dir", type=Path, required=True)
    parser.add_argument("--patch-feature-dir", type=Path, required=True)
    parser.add_argument("--quality-student-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class ScalarRegressionHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 192), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, global_feature, patch_tokens, patch_mask):
        return torch.sigmoid(self.net(global_feature).squeeze(1)), None


class OrdinalWeightedPatchHead(nn.Module):
    """Global ordinal rating plus correlation-weighted local patch ratings."""

    def __init__(self, global_dim, patch_dim, bins=10):
        super().__init__()
        self.threshold_count = bins - 1
        self.global_head = nn.Sequential(nn.LayerNorm(global_dim), nn.Linear(global_dim, 192), nn.GELU(),
                                         nn.Dropout(0.1), nn.Linear(192, self.threshold_count))
        self.patch_head = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, 96), nn.GELU(),
                                        nn.Linear(96, self.threshold_count))
        self.patch_reliability = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, 64), nn.GELU(),
                                               nn.Linear(64, 1))
        self.fusion_gate = nn.Sequential(nn.LayerNorm(global_dim), nn.Linear(global_dim, 32), nn.GELU(),
                                         nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, global_feature, patch_tokens, patch_mask):
        global_logits = self.global_head(global_feature)
        local_logits = self.patch_head(patch_tokens)
        weights = self.patch_reliability(patch_tokens).squeeze(2).masked_fill(~patch_mask, -1e9)
        weights = torch.softmax(weights, 1)
        local_logits = torch.sum(weights[:, :, None] * local_logits, 1)
        gate = self.fusion_gate(global_feature)
        logits = gate * global_logits + (1.0 - gate) * local_logits
        score = torch.sigmoid(logits).mean(1)
        return score, {"ordinal_logits": logits, "patch_weights": weights, "global_gate": gate.squeeze(1)}


def metric_scales(samples):
    scales = {}
    for key, mask_key in (("geometry", "geometry_mask"), ("texture", "texture_mask")):
        values = samples["train"][key][samples["train"][mask_key]]
        scales[key] = torch.quantile(values, 0.95, dim=0).clamp_min(1e-6)
    return scales


def objective_quality_targets(samples, scales):
    result = {}
    geometry_classes = torch.as_tensor(GEOMETRY_CLASSES, device=samples["train"]["attack"].device)
    for split, sample in samples.items():
        severity = sample["severity"]
        geometry_burden = torch.mean(torch.clamp(sample["geometry"] / scales["geometry"], 0.0, 1.0), 1)
        texture_burden = torch.mean(torch.clamp(sample["texture"] / scales["texture"], 0.0, 1.0), 1)
        is_geometry = torch.isin(sample["attack"], geometry_classes)
        is_clean = sample["attack"] == 0
        relevant = torch.where(is_geometry, geometry_burden, texture_burden)
        degradation = 0.5 * severity + 0.5 * relevant
        degradation = torch.where(is_clean, torch.zeros_like(degradation), degradation)
        result[split] = torch.clamp(1.0 - degradation, 0.0, 1.0)
    return result


@torch.no_grad()
def encode_student(student, samples, geometry_stats, texture_stats, scales):
    student.eval()
    encoded = {}
    gm, gs = geometry_stats
    tm, ts = texture_stats
    geometry_indices = torch.as_tensor(GEOMETRY_CLASSES, device=gm.device)
    for split, sample in samples.items():
        out = model_forward(student, sample)
        attack_probability = torch.softmax(out["attack"], 1)
        geometry_raw = torch.clamp(out["geometry"] * gs + gm, min=0.0)
        texture_raw = torch.clamp(out["texture"] * ts + tm, min=0.0)
        geometry_burden = torch.mean(torch.clamp(geometry_raw / scales["geometry"], 0.0, 1.0), 1)
        texture_burden = torch.mean(torch.clamp(texture_raw / scales["texture"], 0.0, 1.0), 1)
        geometry_probability = attack_probability[:, geometry_indices].sum(1)
        clean_probability = attack_probability[:, 0]
        texture_probability = torch.clamp(1.0 - geometry_probability - clean_probability, min=0.0)
        predicted_burden = geometry_probability * geometry_burden + texture_probability * texture_burden
        deterministic = torch.clamp(1.0 - 0.5 * out["severity"] - 0.5 * predicted_burden, 0.0, 1.0)
        global_feature = torch.cat([
            out["quality_embedding"], attack_probability, out["severity"][:, None],
            out["geometry"], out["texture"],
        ], 1)
        encoded[split] = {
            "global": global_feature.detach(), "patch": out["patch_tokens"].detach(),
            "patch_mask": sample["patch_mask"], "deterministic": deterministic.detach(),
            "inverse_severity": (1.0 - out["severity"]).detach(),
        }
    return encoded


def pairwise_rank_loss(prediction, target, minimum_gap=0.05):
    difference = target[:, None] - target[None, :]
    valid = torch.triu(torch.abs(difference) >= minimum_gap, diagonal=1)
    if not valid.any():
        return prediction.sum() * 0.0
    predicted_difference = prediction[:, None] - prediction[None, :]
    signed = torch.sign(difference[valid]) * predicted_difference[valid]
    return F.softplus(-signed / 0.1).mean()


def metrics(prediction, target):
    pred = prediction.detach().cpu().numpy()
    truth = target.detach().cpu().numpy()
    differences = truth[:, None] - truth[None, :]
    valid = np.triu(np.abs(differences) >= 0.05, 1)
    pairwise = float(np.mean((np.sign(pred[:, None] - pred[None, :])[valid]
                              == np.sign(differences[valid])))) if valid.any() else 0.0
    mae = float(np.mean(np.abs(pred - truth)))
    return {
        "mae_0_1": mae, "mae_0_100": 100.0 * mae,
        "plcc": correlation(pred, truth),
        "srcc": correlation(rankdata(pred), rankdata(truth)),
        "pairwise_accuracy": pairwise,
        "within_10_points": float(np.mean(np.abs(pred - truth) <= 0.10)),
        "predicted_mean_0_100": float(100.0 * pred.mean()),
        "target_mean_0_100": float(100.0 * truth.mean()),
    }


def build_monotonic_pairs(feature_dir, split):
    with np.load(feature_dir / f"scores_{split}.npz") as values:
        asset_ids = values["query_ids"].astype(str)
        attacks = values["attacks"].astype(str)
        severities = values["severities"].astype(np.float32)
        gallery_count = len(values["gallery_point"])
    groups = {}
    for query_index, (asset_id, attack, severity) in enumerate(zip(asset_ids, attacks, severities)):
        groups.setdefault((asset_id, attack), []).append((float(severity), gallery_count + query_index))
    pairs = []
    for levels in groups.values():
        levels.sort()
        for low_index, low in enumerate(levels):
            for high in levels[low_index + 1:]:
                if high[0] > low[0]:
                    pairs.append((low[1], high[1]))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def monotonic_accuracy(prediction, pairs):
    if len(pairs) == 0:
        return 0.0
    index = torch.as_tensor(pairs, device=prediction.device)
    return float((prediction[index[:, 0]] >= prediction[index[:, 1]]).float().mean())


@torch.no_grad()
def evaluate(model, encoded, targets, monotonic_pairs=None):
    model.eval()
    score, aux = model(encoded["global"], encoded["patch"], encoded["patch_mask"])
    result = metrics(score, targets)
    if aux is not None:
        result["mean_global_gate"] = float(aux["global_gate"].mean())
        result["mean_patch_entropy"] = float(torch.mean(
            -torch.sum(aux["patch_weights"] * torch.log(aux["patch_weights"].clamp_min(1e-8)), 1)))
    if monotonic_pairs is not None:
        result["severity_monotonic_accuracy"] = monotonic_accuracy(score, monotonic_pairs)
    return result


@torch.no_grad()
def predict(model, encoded):
    model.eval()
    return model(encoded["global"], encoded["patch"], encoded["patch_mask"])[0]


def train_head(kind, encoded, targets, monotonic_pairs, epochs, seed, device):
    seed_all(seed)
    global_dim = encoded["train"]["global"].shape[1]
    if kind == "scalar_regression":
        model = ScalarRegressionHead(global_dim).to(device)
    else:
        model = OrdinalWeightedPatchHead(global_dim, encoded["train"]["patch"].shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    best, best_state, history = None, None, []
    train = encoded["train"]
    thresholds = torch.arange(1, 10, device=device).float() / 10.0
    ordinal_truth = (targets["train"][:, None] > thresholds[None, :]).float()
    positive = ordinal_truth.sum(0).clamp_min(1.0)
    pos_weight = ((len(ordinal_truth) - positive) / positive).clamp(0.25, 4.0)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(targets["train"]), device=device)
        losses = []
        for start in range(0, len(order), 128):
            index = order[start:start + 128]
            score, aux = model(train["global"][index], train["patch"][index], train["patch_mask"][index])
            if kind == "scalar_regression":
                loss = F.smooth_l1_loss(score, targets["train"][index])
            else:
                loss = F.binary_cross_entropy_with_logits(
                    aux["ordinal_logits"], ordinal_truth[index], pos_weight=pos_weight)
                probability = torch.sigmoid(aux["ordinal_logits"])
                loss = loss + 0.05 * F.relu(probability[:, 1:] - probability[:, :-1]).mean()
            loss = loss + 0.20 * pairwise_rank_loss(score, targets["train"][index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if len(monotonic_pairs["train"]):
            pair_index = torch.as_tensor(monotonic_pairs["train"], device=device)
            low, _ = model(train["global"][pair_index[:, 0]], train["patch"][pair_index[:, 0]],
                           train["patch_mask"][pair_index[:, 0]])
            high, _ = model(train["global"][pair_index[:, 1]], train["patch"][pair_index[:, 1]],
                            train["patch_mask"][pair_index[:, 1]])
            monotonic_loss = (F.softplus(5.0 * (high - low + 0.02)) / 5.0).mean()
            optimizer.zero_grad(set_to_none=True); (0.10 * monotonic_loss).backward(); optimizer.step()
        val = evaluate(model, encoded["val"], targets["val"], monotonic_pairs["val"])
        selection_error = (val["mae_0_1"] + 0.25 * (1.0 - val["srcc"])
                           + 0.10 * (1.0 - val["severity_monotonic_accuracy"]))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "selection_error": selection_error, **val})
        key = (selection_error, epoch)
        if best is None or key < best:
            best, best_state = key, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    results = {split: evaluate(model, encoded[split], targets[split], monotonic_pairs[split])
               for split in ("val", "test", "blind")}
    return model, {"name": kind, "best_epoch": min(history, key=lambda x: (x["selection_error"], x["epoch"]))["epoch"],
                   "results": results, "history": history}


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required: formal quality-index training must run on the rental GPU")
    store = FeatureStore(args.feature_dir, device)
    samples, geometry_names, texture_names = build_samples(
        store, args.feature_dir, args.geometry_target_dir, args.texture_target_dir, device)
    patch_dim = attach_patch_features(samples, args.patch_feature_dir, device)
    geometry_stats = stats(samples, "geometry", "geometry_mask")
    texture_stats = stats(samples, "texture", "texture_mask")
    scales = metric_scales(samples)
    targets = objective_quality_targets(samples, scales)
    checkpoint = torch.load(args.quality_student_checkpoint, map_location=device, weights_only=True)
    selected_name = checkpoint.get("selected_variant", "local_patch_shared")
    geometry_only = "geometry_only" in selected_name
    student = NoReferenceQualityStudent(
        store.dims, len(geometry_names), len(texture_names), patch_dim=patch_dim,
        patch_geometry_only=geometry_only,
        modality_aware=bool(checkpoint.get("modality_aware", False))).to(device)
    student.load_state_dict(checkpoint["model"])
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    encoded = encode_student(student, samples, geometry_stats, texture_stats, scales)
    monotonic_pairs = {split: build_monotonic_pairs(args.feature_dir, split)
                       for split in ("train", "val", "test", "blind")}

    baselines = {
        "inverse_severity": {split: metrics(encoded[split]["inverse_severity"], targets[split])
                             for split in ("val", "test", "blind")},
        "deterministic_multitask": {split: metrics(encoded[split]["deterministic"], targets[split])
                                    for split in ("val", "test", "blind")},
    }
    for baseline_name, prediction_name in (("inverse_severity", "inverse_severity"),
                                              ("deterministic_multitask", "deterministic")):
        for split in ("val", "test", "blind"):
            baselines[baseline_name][split]["severity_monotonic_accuracy"] = monotonic_accuracy(
                encoded[split][prediction_name], monotonic_pairs[split])
    variants, models = [], {}
    for kind in ("scalar_regression", "ordinal_weighted_patch"):
        model, result = train_head(kind, encoded, targets, monotonic_pairs, args.epochs, args.seed, device)
        models[kind], variants = model, variants + [result]
    ordinal_predictions = {split: predict(models["ordinal_weighted_patch"], encoded[split])
                           for split in ("val", "test", "blind")}
    fusion_trials = []
    for ordinal_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        validation_prediction = (ordinal_weight * ordinal_predictions["val"]
                                 + (1.0 - ordinal_weight) * encoded["val"]["deterministic"])
        validation = metrics(validation_prediction, targets["val"])
        validation["severity_monotonic_accuracy"] = monotonic_accuracy(
            validation_prediction, monotonic_pairs["val"])
        fusion_trials.append({"ordinal_weight": ordinal_weight, "validation": validation,
                              "selection_error": (validation["mae_0_1"]
                                  + 0.25 * (1.0 - validation["srcc"])
                                  + 0.10 * (1.0 - validation["severity_monotonic_accuracy"]))})
    selected_weight = min(fusion_trials, key=lambda x: (x["selection_error"], x["ordinal_weight"]))["ordinal_weight"]
    fused_results = {}
    for split in ("val", "test", "blind"):
        prediction = (selected_weight * ordinal_predictions[split]
                      + (1.0 - selected_weight) * encoded[split]["deterministic"])
        fused_results[split] = metrics(prediction, targets[split])
        fused_results[split]["severity_monotonic_accuracy"] = monotonic_accuracy(
            prediction, monotonic_pairs[split])
    deployment_candidates = {
        "deterministic_multitask": baselines["deterministic_multitask"],
        **{row["name"]: row["results"] for row in variants},
        "validation_fused": fused_results,
    }
    selected_name = min(deployment_candidates, key=lambda name: (
        deployment_candidates[name]["val"]["mae_0_1"]
        + 0.25 * (1.0 - deployment_candidates[name]["val"]["srcc"])
        + 0.10 * (1.0 - deployment_candidates[name]["val"].get("severity_monotonic_accuracy", 0.0)), name))
    output = {
        "status": "OBJECTIVE_QUALITY_INDEX_COMPLETE", "seed": args.seed,
        "device": torch.cuda.get_device_name(device), "selected_variant": selected_name,
        "protocol": {
            "inference": "single unseen mesh; no pristine reference",
            "index_range": [0, 100], "higher_is_better": True,
            "target_definition": "100 * (1 - 0.5 normalized attack severity - 0.5 relevant measured geometry/texture burden)",
            "not_mos": True, "identity_disjoint_splits": True,
            "selection": "validation MAE + 0.25*(1-SRCC) + 0.10*(1-severity monotonic accuracy); test/blind locked",
            "identity_attack_monotonic_loss_weight": 0.10,
            "identity_attack_monotonic_pairs": "all ordered severity pairs within the same identity and attack",
            "borrowed_ideas": ["COPP-Net correlation-weighted patch prediction",
                               "ordinal regression with continuous inference", "relative ranking loss"],
        },
        "metric_scales_p95": {key: value.cpu().tolist() for key, value in scales.items()},
        "baselines": baselines, "variants": variants,
        "fusion": {"selected_ordinal_weight": selected_weight, "trials": fusion_trials,
                   "results": fused_results},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": models["ordinal_weighted_patch"].state_dict(), "selected_variant": selected_name,
                "ordinal_weight": selected_weight,
                "global_dim": encoded["train"]["global"].shape[1], "patch_dim": patch_dim,
                "seed": args.seed}, args.output_dir / "quality_index_head.pt")
    (args.output_dir / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"selected_variant": selected_name, "selected_ordinal_weight": selected_weight,
               "baselines": baselines, "variants": {row["name"]: row["results"] for row in variants},
               "fused": fused_results}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
