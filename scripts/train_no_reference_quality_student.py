#!/usr/bin/env python3
"""Train a single-model no-reference quality student from the full-reference teacher."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_branch_aware_multitask_quality import (
    ATTACKS,
    BranchAwareQuality,
    build_samples,
    correlation,
    rankdata,
    stats,
    vector_metrics,
)
from train_four_branch_fusion import FeatureStore
from train_frozen_base_quality_head import seed_all


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--geometry-target-dir", type=Path, required=True)
    parser.add_argument("--texture-target-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--patch-feature-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class NoReferenceQualityStudent(nn.Module):
    """Four branch tokens, cross-branch attention and task-specific confidence gates."""

    def __init__(self, dims, geometry_dim, texture_dim, patch_dim=None, patch_geometry_only=False,
                 modality_aware=False):
        super().__init__()
        self.uses_patches = patch_dim is not None
        self.patch_geometry_only = bool(patch_geometry_only)
        self.modality_aware = bool(modality_aware)
        self.branch = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
            for dim in dims
        ])
        if self.uses_patches:
            self.patch_encoder = nn.Sequential(
                nn.LayerNorm(patch_dim), nn.Linear(patch_dim, 128), nn.GELU(),
                nn.Linear(128, 128), nn.GELU())
            self.patch_attention = nn.MultiheadAttention(128, 4, dropout=0.1, batch_first=True)
            self.patch_norm = nn.LayerNorm(128)
            self.patch_pool = nn.Linear(128, 1)
        token_count = 5 if self.uses_patches and not self.patch_geometry_only else 4
        self.branch_embedding = nn.Parameter(torch.zeros(1, token_count, 128))
        nn.init.normal_(self.branch_embedding, std=0.02)
        if self.modality_aware:
            self.missing_texture_token = nn.Parameter(torch.zeros(1, 128))
            nn.init.normal_(self.missing_texture_token, std=0.02)
        self.attention = nn.MultiheadAttention(128, 4, dropout=0.1, batch_first=True)
        self.attention_norm = nn.LayerNorm(128)
        self.shared = nn.Sequential(nn.Linear(128 * token_count, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1))
        self.attack = nn.Linear(256, len(ATTACKS))
        self.severity = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.geometry_gate = nn.Linear(128, 1)
        self.texture_gate = nn.Linear(128, 1)
        geometry_input = 256 if self.uses_patches and self.patch_geometry_only else 128
        self.geometry = nn.Sequential(nn.Linear(geometry_input, 128), nn.GELU(), nn.Linear(128, geometry_dim))
        self.texture = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, texture_dim))

    def forward(self, query, patches=None, patch_mask=None, texture_available=None):
        tokens = [layer(branch) for layer, branch in zip(self.branch, query)]
        if self.modality_aware and texture_available is not None:
            available = texture_available.to(dtype=torch.bool, device=tokens[3].device).reshape(-1, 1)
            missing = self.missing_texture_token.expand(len(tokens[3]), -1)
            tokens[3] = torch.where(available, tokens[3], missing)
        patch_global = None
        if self.uses_patches:
            patch_tokens = self.patch_encoder(patches)
            attended, _ = self.patch_attention(
                patch_tokens, patch_tokens, patch_tokens,
                key_padding_mask=~patch_mask, need_weights=False)
            patch_tokens = self.patch_norm(patch_tokens + attended)
            logits = self.patch_pool(patch_tokens).squeeze(2).masked_fill(~patch_mask, -1e9)
            patch_global = torch.sum(torch.softmax(logits, 1)[:, :, None] * patch_tokens, 1)
            if not self.patch_geometry_only:
                tokens.append(patch_global)
        tokens = torch.stack(tokens, 1)
        tokens = tokens + self.branch_embedding
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.attention_norm(tokens + attended)
        shared = self.shared(tokens.flatten(1))
        geometry_weights = torch.softmax(self.geometry_gate(tokens).squeeze(2), 1)
        texture_weights = torch.softmax(self.texture_gate(tokens).squeeze(2), 1)
        geometry_feature = torch.sum(geometry_weights[:, :, None] * tokens, 1)
        texture_feature = torch.sum(texture_weights[:, :, None] * tokens, 1)
        geometry_input = (torch.cat([geometry_feature, patch_global], 1)
                          if self.uses_patches and self.patch_geometry_only else geometry_feature)
        return {
            "attack": self.attack(shared),
            "severity": self.severity(shared).squeeze(1),
            "geometry": self.geometry(geometry_input),
            "texture": self.texture(texture_feature),
            "geometry_weights": geometry_weights,
            "texture_weights": texture_weights,
            "quality_embedding": shared,
            "geometry_embedding": geometry_feature,
            "patch_tokens": patch_tokens if self.uses_patches else None,
            "patch_weights": (torch.softmax(logits, 1) if self.uses_patches else None),
        }


def supervised_contrastive_loss(features, labels, temperature=0.1):
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    logits = logits.masked_fill(diagonal, -1e9)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positives.sum(1)
    valid = positive_count > 0
    if not valid.any():
        return features.sum() * 0.0
    return -(log_probability * positives).sum(1)[valid].div(positive_count[valid]).mean()


def macro_f1(predicted, truth):
    values = []
    for index in range(len(ATTACKS)):
        tp = int(((predicted == index) & (truth == index)).sum())
        fp = int(((predicted == index) & (truth != index)).sum())
        fn = int(((predicted != index) & (truth == index)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(values))


def model_forward(model, sample, index=None, texture_available=None, query_override=None):
    query = (sample["degraded"] if index is None else [x[index] for x in sample["degraded"]])
    if query_override is not None:
        query = query_override
    if not model.uses_patches:
        return model(query, texture_available=texture_available)
    patches = sample["patch"] if index is None else sample["patch"][index]
    mask = sample["patch_mask"] if index is None else sample["patch_mask"][index]
    return model(query, patches, mask, texture_available=texture_available)


@torch.no_grad()
def evaluate(model, sample, geometry_stats, texture_stats, geometry_names, texture_names, severity_baseline):
    model.eval()
    out = model_forward(model, sample)
    predicted, truth = out["attack"].argmax(1), sample["attack"]
    severity_true = sample["severity"].cpu().numpy()
    severity_pred = out["severity"].cpu().numpy()
    gm, gs = geometry_stats
    tm, ts = texture_stats
    geometry_prediction = out["geometry"][sample["geometry_mask"]] * gs + gm
    texture_prediction = out["texture"][sample["texture_mask"]] * ts + tm
    geometry = vector_metrics(geometry_prediction, sample["geometry"][sample["geometry_mask"]], gm, geometry_names)
    texture = vector_metrics(texture_prediction, sample["texture"][sample["texture_mask"]], tm, texture_names)
    severity_mae = float(np.mean(np.abs(severity_pred - severity_true)))
    result = {
        "count": len(truth),
        "accuracy": float((predicted == truth).float().mean()),
        "macro_f1": macro_f1(predicted, truth),
        "severity_mae": severity_mae,
        "severity_normalized_mae": severity_mae / max(severity_baseline, 1e-12),
        "severity_srcc": correlation(rankdata(severity_pred), rankdata(severity_true)),
        "geometry": geometry,
        "texture": texture,
        "mean_geometry_weights": out["geometry_weights"].mean(0).cpu().tolist(),
        "mean_texture_weights": out["texture_weights"].mean(0).cpu().tolist(),
    }
    result["selection_error"] = float(np.mean([
        1.0 - result["macro_f1"], result["severity_normalized_mae"],
        geometry["mean_normalized_mae"], texture["mean_normalized_mae"],
    ]))
    return result


def train_variant(name, distillation_weight, contrastive_weight, contrastive_target,
                  geometry_loss_weight, use_patches, patch_geometry_only,
                  store, samples, teacher, geometry_stats, texture_stats,
                  geometry_names, texture_names, severity_baseline, epochs, seed, device):
    seed_all(seed)
    patch_dim = samples["train"]["patch"].shape[-1] if use_patches else None
    model = NoReferenceQualityStudent(
        store.dims, len(geometry_names), len(texture_names), patch_dim=patch_dim,
        patch_geometry_only=patch_geometry_only).to(device)
    train = samples["train"]
    gm, gs = geometry_stats
    tm, ts = texture_stats
    counts = torch.bincount(train["attack"], minlength=len(ATTACKS)).float()
    class_weights = counts.sum() / torch.clamp(counts * len(ATTACKS), min=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    best, best_state, history = None, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train["attack"]), device=device)
        epoch_losses = []
        for start in range(0, len(order), 128):
            index = order[start:start + 128]
            query = [x[index] for x in train["degraded"]]
            out = model_forward(model, train, index)
            loss = F.cross_entropy(out["attack"], train["attack"][index], weight=class_weights)
            loss = loss + 0.5 * F.mse_loss(out["severity"], train["severity"][index])
            gmask, tmask = train["geometry_mask"][index], train["texture_mask"][index]
            if gmask.any():
                loss = loss + geometry_loss_weight * F.smooth_l1_loss(
                    out["geometry"][gmask], (train["geometry"][index][gmask] - gm) / gs)
            if tmask.any():
                loss = loss + 0.5 * F.smooth_l1_loss(out["texture"][tmask], (train["texture"][index][tmask] - tm) / ts)
            if contrastive_weight > 0:
                if contrastive_target == "geometry":
                    geometry_classes = torch.as_tensor([0, 5, 7, 10], device=device)
                    selected = torch.isin(train["attack"][index], geometry_classes)
                    contrast = supervised_contrastive_loss(
                        out["geometry_embedding"][selected], train["attack"][index][selected])
                else:
                    contrast = supervised_contrastive_loss(
                        out["quality_embedding"], train["attack"][index])
                loss = loss + contrastive_weight * contrast
            if distillation_weight > 0:
                with torch.no_grad():
                    teacher_out = teacher([x[index] for x in train["reference"]], query)
                temperature = 2.0
                distill = F.kl_div(
                    F.log_softmax(out["attack"] / temperature, dim=1),
                    F.softmax(teacher_out["attack"] / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature * temperature
                distill = distill + 0.5 * F.mse_loss(out["severity"], teacher_out["severity"])
                if gmask.any():
                    distill = distill + 0.5 * F.smooth_l1_loss(out["geometry"][gmask], teacher_out["geometry"][gmask])
                if tmask.any():
                    distill = distill + 0.5 * F.smooth_l1_loss(out["texture"][tmask], teacher_out["texture"][tmask])
                loss = loss + distillation_weight * distill
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        val = evaluate(model, samples["val"], geometry_stats, texture_stats, geometry_names, texture_names, severity_baseline)
        history.append({"epoch": epoch, "loss": float(np.mean(epoch_losses)), "selection_error": val["selection_error"],
                        "macro_f1": val["macro_f1"], "severity_mae": val["severity_mae"],
                        "geometry_nmae": val["geometry"]["mean_normalized_mae"],
                        "texture_nmae": val["texture"]["mean_normalized_mae"]})
        key = (val["selection_error"], epoch)
        if best is None or key < best:
            best, best_state = key, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    results = {split: evaluate(model, samples[split], geometry_stats, texture_stats, geometry_names,
                               texture_names, severity_baseline) for split in ("val", "test", "blind")}
    return model, {
        "name": name,
        "distillation_weight": distillation_weight,
        "contrastive_weight": contrastive_weight,
        "contrastive_target": contrastive_target,
        "geometry_loss_weight": geometry_loss_weight,
        "uses_local_mesh_patches": use_patches,
        "patch_geometry_only": patch_geometry_only,
        "best_epoch": min(history, key=lambda x: (x["selection_error"], x["epoch"]))["epoch"],
        "results": results,
        "history": history,
    }


def compact(result):
    return {split: {
        "macro_f1": values["macro_f1"],
        "severity_mae": values["severity_mae"],
        "geometry_nmae": values["geometry"]["mean_normalized_mae"],
        "texture_nmae": values["texture"]["mean_normalized_mae"],
        "selection_error": values["selection_error"],
    } for split, values in result["results"].items()}


def attach_patch_features(samples, root, device):
    raw = {}
    for split in ("train", "val", "test", "blind"):
        with np.load(root / f"patches_{split}.npz") as values:
            raw[split] = {key: values[key].copy() for key in values.files}
    valid_train = raw["train"]["gallery_patch"][raw["train"]["gallery_mask"]]
    valid_train_query = raw["train"]["query_patch"][raw["train"]["query_mask"]]
    valid_train = np.concatenate([valid_train, valid_train_query], 0)
    mean = valid_train.mean(0, keepdims=True)
    std = np.maximum(valid_train.std(0, keepdims=True), 1e-5)
    for split, values in raw.items():
        patch = np.concatenate([values["gallery_patch"], values["query_patch"]], 0).astype(np.float32)
        mask = np.concatenate([values["gallery_mask"], values["query_mask"]], 0).astype(bool)
        patch = (patch - mean) / std
        patch[~mask] = 0.0
        if len(patch) != len(samples[split]["attack"]):
            raise ValueError(f"{split}: patch and quality sample counts differ")
        samples[split]["patch"] = torch.as_tensor(patch, device=device)
        samples[split]["patch_mask"] = torch.as_tensor(mask, device=device)
    return int(mean.shape[-1])


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required: formal experiments must run on the rental GPU")
    store = FeatureStore(args.feature_dir, device)
    samples, geometry_names, texture_names = build_samples(
        store, args.feature_dir, args.geometry_target_dir, args.texture_target_dir, device)
    if args.patch_feature_dir is None:
        raise ValueError("--patch-feature-dir is required for the local-patch ablation")
    patch_dim = attach_patch_features(samples, args.patch_feature_dir, device)
    geometry_stats = stats(samples, "geometry", "geometry_mask")
    texture_stats = stats(samples, "texture", "texture_mask")
    severity_mean = samples["train"]["severity"].mean()
    severity_baseline = float(torch.mean(torch.abs(samples["val"]["severity"] - severity_mean)))
    teacher = BranchAwareQuality(store.dims, len(geometry_names), len(texture_names)).to(device)
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=True)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    variants = []
    models = {}
    for name, distill_weight, contrast_weight, contrast_target, geometry_weight, use_patches, geometry_only in (
        ("four_branch_quality_contrastive", 0.0, 0.08, "shared", 0.5, False, False),
        ("local_patch_shared", 0.0, 0.0, "none", 0.5, True, False),
        ("local_patch_geometry_only", 0.0, 0.0, "none", 0.5, True, True),
    ):
        model, result = train_variant(
            name, distill_weight, contrast_weight, contrast_target, geometry_weight,
            use_patches, geometry_only,
            store, samples, teacher, geometry_stats, texture_stats,
            geometry_names, texture_names, severity_baseline, args.epochs, args.seed, device)
        models[name], variants = model, variants + [result]
    selected = min(variants, key=lambda item: (item["results"]["val"]["selection_error"], item["name"]))
    output = {
        "status": "NO_REFERENCE_QUALITY_STUDENT_COMPLETE",
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device),
        "selected_variant": selected["name"],
        "protocol": {
            "base_frozen": True,
            "training_teacher_full_reference": True,
            "inference_full_reference": False,
            "inference_input": "one unseen mesh represented by four frozen Base branches",
            "identity_disjoint_splits": True,
            "test_blind_locked": True,
            "ablation": "four global branches vs local mesh patches vs local patches plus quality-aware contrastive learning",
            "local_patch_descriptor_dim": patch_dim,
            "selection": "minimum validation mean of classification error, normalized severity MAE, geometry NMAE, texture NMAE",
        },
        "geometry_metrics": geometry_names,
        "texture_metrics": texture_names,
        "variants": variants,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": models[selected["name"]].state_dict(), "dims": store.dims,
                "selected_variant": selected["name"], "seed": args.seed}, args.output_dir / "quality_student.pt")
    (args.output_dir / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_variant": selected["name"],
                      "variants": {item["name"]: compact(item) for item in variants}}, indent=2))


if __name__ == "__main__":
    main()
