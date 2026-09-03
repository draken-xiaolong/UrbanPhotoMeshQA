#!/usr/bin/env python3
"""Train a no-reference quality student robust to missing texture modalities."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_branch_aware_multitask_quality import ATTACKS, build_samples, stats, vector_metrics
from train_four_branch_fusion import FeatureStore
from train_frozen_base_quality_head import seed_all
from train_no_reference_quality_student import (
    NoReferenceQualityStudent, attach_patch_features, evaluate, macro_f1, model_forward,
)

GEOMETRY_CLASSES = (0, 5, 7, 10)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--geometry-target-dir", type=Path, required=True)
    parser.add_argument("--texture-target-dir", type=Path, required=True)
    parser.add_argument("--patch-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--texture-dropout", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def evaluate_missing_texture(model, sample, geometry_stats, geometry_names, severity_baseline):
    model.eval()
    selected = torch.isin(sample["attack"], torch.as_tensor(GEOMETRY_CLASSES, device=sample["attack"].device))
    index = torch.nonzero(selected, as_tuple=False).flatten()
    query = [branch[index].clone() for branch in sample["degraded"]]
    query[3].zero_()
    available = torch.zeros(len(index), dtype=torch.bool, device=index.device)
    out = model_forward(model, sample, index, texture_available=available, query_override=query)
    truth = sample["attack"][index]
    predicted = out["attack"].argmax(1)
    severity_mae = float(torch.mean(torch.abs(out["severity"] - sample["severity"][index])))
    gm, gs = geometry_stats
    gmask = sample["geometry_mask"][index]
    geometry_prediction = out["geometry"][gmask] * gs + gm
    geometry = vector_metrics(
        geometry_prediction, sample["geometry"][index][gmask], gm, geometry_names)
    return {
        "count": int(len(index)),
        "attack_accuracy": float((predicted == truth).float().mean()),
        "macro_f1_all_classes": macro_f1(predicted, truth),
        "severity_mae": severity_mae,
        "severity_normalized_mae": severity_mae / max(severity_baseline, 1e-12),
        "geometry_nmae": geometry["mean_normalized_mae"],
    }


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required: formal experiments must run on the rental GPU")
    store = FeatureStore(args.feature_dir, device)
    samples, geometry_names, texture_names = build_samples(
        store, args.feature_dir, args.geometry_target_dir, args.texture_target_dir, device)
    patch_dim = attach_patch_features(samples, args.patch_feature_dir, device)
    geometry_stats = stats(samples, "geometry", "geometry_mask")
    texture_stats = stats(samples, "texture", "texture_mask")
    severity_mean = samples["train"]["severity"].mean()
    severity_baseline = float(torch.mean(torch.abs(samples["val"]["severity"] - severity_mean)))
    model = NoReferenceQualityStudent(
        store.dims, len(geometry_names), len(texture_names), patch_dim=patch_dim,
        modality_aware=True).to(device)
    train = samples["train"]
    gm, gs = geometry_stats
    tm, ts = texture_stats
    counts = torch.bincount(train["attack"], minlength=len(ATTACKS)).float()
    class_weights = counts.sum() / torch.clamp(counts * len(ATTACKS), min=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    best, best_state, history = None, None, []
    geometry_classes = torch.as_tensor(GEOMETRY_CLASSES, device=device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train["attack"]), device=device)
        losses = []
        for start in range(0, len(order), 128):
            index = order[start:start + 128]
            query = [branch[index].clone() for branch in train["degraded"]]
            eligible = torch.isin(train["attack"][index], geometry_classes)
            dropped = eligible & (torch.rand(len(index), device=device) < args.texture_dropout)
            query[3][dropped] = 0.0
            out = model_forward(
                model, train, index, texture_available=~dropped, query_override=query)
            loss = F.cross_entropy(out["attack"], train["attack"][index], weight=class_weights)
            loss = loss + 0.5 * F.mse_loss(out["severity"], train["severity"][index])
            gmask, tmask = train["geometry_mask"][index], train["texture_mask"][index] & ~dropped
            if gmask.any():
                loss = loss + 0.5 * F.smooth_l1_loss(
                    out["geometry"][gmask], (train["geometry"][index][gmask] - gm) / gs)
            if tmask.any():
                loss = loss + 0.5 * F.smooth_l1_loss(
                    out["texture"][tmask], (train["texture"][index][tmask] - tm) / ts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        normal = evaluate(
            model, samples["val"], geometry_stats, texture_stats,
            geometry_names, texture_names, severity_baseline)
        missing = evaluate_missing_texture(
            model, samples["val"], geometry_stats, geometry_names, severity_baseline)
        missing_error = np.mean([
            1.0 - missing["attack_accuracy"], missing["severity_normalized_mae"], missing["geometry_nmae"]])
        selection_error = normal["selection_error"] + 0.25 * float(missing_error)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "selection_error": selection_error, "normal": normal, "missing_texture": missing})
        key = (selection_error, epoch)
        if best is None or key < best:
            best, best_state = key, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    results = {}
    for split in ("val", "test", "blind"):
        results[split] = {
            "normal": evaluate(model, samples[split], geometry_stats, texture_stats,
                               geometry_names, texture_names, severity_baseline),
            "missing_texture_geometry_subset": evaluate_missing_texture(
                model, samples[split], geometry_stats, geometry_names, severity_baseline),
        }
    output = {
        "status": "MODALITY_AWARE_QUALITY_STUDENT_COMPLETE", "seed": args.seed,
        "device": torch.cuda.get_device_name(device), "selected_variant": "local_patch_modality_aware",
        "protocol": {"texture_dropout": args.texture_dropout,
                     "dropout_scope": "clean and geometry-degraded samples only",
                     "missing_texture_token": True, "identity_disjoint_splits": True},
        "best_epoch": min(history, key=lambda row: (row["selection_error"], row["epoch"]))["epoch"],
        "geometry_metrics": geometry_names, "texture_metrics": texture_names,
        "results": results, "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "dims": store.dims,
                "selected_variant": "local_patch_modality_aware", "modality_aware": True,
                "seed": args.seed}, args.output_dir / "quality_student.pt")
    (args.output_dir / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": output["best_epoch"], "results": results}, indent=2))


if __name__ == "__main__":
    main()
