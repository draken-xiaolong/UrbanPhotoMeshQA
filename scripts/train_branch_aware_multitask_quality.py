#!/usr/bin/env python3
"""Train a frozen-Base branch-aware full-reference multi-task quality head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_four_branch_fusion import FeatureStore
from train_frozen_base_quality_head import ATTACKS, correlation, make_labels, rankdata, seed_all


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--geometry-target-dir", type=Path, required=True)
    parser.add_argument("--texture-target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class BranchAwareQuality(nn.Module):
    def __init__(self, dims, geometry_dim, texture_dim):
        super().__init__()
        self.branch = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim * 2 + 1), nn.Linear(dim * 2 + 1, 128),
                          nn.GELU(), nn.Linear(128, 128), nn.GELU()) for dim in dims
        ])
        self.shared = nn.Sequential(nn.Linear(128 * 4, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1))
        self.attack = nn.Linear(256, len(ATTACKS))
        self.severity = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.geometry_gate = nn.Linear(128, 1)
        self.texture_gate = nn.Linear(128, 1)
        self.geometry = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, geometry_dim))
        self.texture = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, texture_dim))

    def forward(self, reference, degraded):
        tokens = []
        for layer, clean, query in zip(self.branch, reference, degraded):
            cosine = F.cosine_similarity(clean, query, dim=1, eps=1e-6).unsqueeze(1)
            tokens.append(layer(torch.cat([torch.abs(clean - query), clean * query, cosine], 1)))
        tokens = torch.stack(tokens, 1)
        shared = self.shared(tokens.flatten(1))
        geometry_weights = torch.softmax(self.geometry_gate(tokens).squeeze(2), 1)
        texture_weights = torch.softmax(self.texture_gate(tokens).squeeze(2), 1)
        geometry_feature = torch.sum(geometry_weights[:, :, None] * tokens, 1)
        texture_feature = torch.sum(texture_weights[:, :, None] * tokens, 1)
        return {
            "attack": self.attack(shared), "severity": self.severity(shared).squeeze(1),
            "geometry": self.geometry(geometry_feature), "texture": self.texture(texture_feature),
            "geometry_weights": geometry_weights, "texture_weights": texture_weights,
        }


def target_lookup(path: Path):
    with np.load(path) as values:
        return ({int(index): row for index, row in zip(values["query_indices"], values["targets"])},
                values["metrics"].astype(str).tolist())


def build_samples(store, feature_dir, geometry_dir, texture_dir, device):
    output, geometry_names, texture_names = {}, None, None
    for split in ("train", "val", "test", "blind"):
        data = store.data[split]
        with np.load(feature_dir / f"scores_{split}.npz") as values:
            attacks = values["attacks"].astype(str); severities = values["severities"].astype(np.float32)
            gallery_index = values["targets"].astype(np.int64)
        geometry_map, geometry_names = target_lookup(geometry_dir / f"targets_{split}.npz")
        texture_map, texture_names = target_lookup(texture_dir / f"targets_{split}.npz")
        gallery_count, query_count = len(data["gallery"][0]), len(data["query"][0])
        reference = [torch.cat([branch, branch[torch.as_tensor(gallery_index, device=device)]])
                     for branch in data["gallery"]]
        degraded = [torch.cat([branch, query]) for branch, query in zip(data["gallery"], data["query"])]
        names = np.concatenate([np.full(gallery_count, "clean"), attacks])
        raw_severity = np.concatenate([np.zeros(gallery_count, np.float32), severities])
        attack_labels, severity_labels = make_labels(names, raw_severity)
        geometry = np.zeros((gallery_count + query_count, len(geometry_names)), np.float32)
        texture = np.zeros((gallery_count + query_count, len(texture_names)), np.float32)
        geometry_mask = np.zeros(gallery_count + query_count, bool); texture_mask = np.zeros_like(geometry_mask)
        geometry_mask[:gallery_count] = True; texture_mask[:gallery_count] = True
        for index, target in geometry_map.items(): geometry[gallery_count + index] = target; geometry_mask[gallery_count + index] = True
        for index, target in texture_map.items(): texture[gallery_count + index] = target; texture_mask[gallery_count + index] = True
        output[split] = {
            "reference": reference, "degraded": degraded,
            "attack": torch.as_tensor(attack_labels, device=device),
            "severity": torch.as_tensor(severity_labels, device=device),
            "geometry": torch.as_tensor(geometry, device=device),
            "texture": torch.as_tensor(texture, device=device),
            "geometry_mask": torch.as_tensor(geometry_mask, device=device),
            "texture_mask": torch.as_tensor(texture_mask, device=device),
        }
    return output, geometry_names, texture_names


def stats(samples, key, mask_key):
    values = samples["train"][key][samples["train"][mask_key]]
    return values.mean(0), values.std(0).clamp_min(1e-8)


def vector_metrics(prediction, target, mean, names):
    pred = prediction.detach().cpu().numpy(); truth = target.detach().cpu().numpy(); center = mean.cpu().numpy()
    result = {}
    for i, name in enumerate(names):
        mae = float(np.mean(np.abs(pred[:, i] - truth[:, i]))); baseline = float(np.mean(np.abs(truth[:, i] - center[i])))
        result[name] = {"mae": mae, "baseline_mae": baseline, "normalized_mae": mae / max(baseline, 1e-12),
                        "plcc": correlation(pred[:, i], truth[:, i]),
                        "srcc": correlation(rankdata(pred[:, i]), rankdata(truth[:, i]))}
    return {"mean_normalized_mae": float(np.mean([v["normalized_mae"] for v in result.values()])), "metrics": result}


@torch.no_grad()
def evaluate(model, sample, geometry_stats, texture_stats, geometry_names, texture_names, severity_baseline):
    model.eval(); out = model(sample["reference"], sample["degraded"])
    predicted = out["attack"].argmax(1); truth = sample["attack"]
    f1 = []
    for index in range(len(ATTACKS)):
        tp = int(((predicted == index) & (truth == index)).sum()); fp = int(((predicted == index) & (truth != index)).sum()); fn = int(((predicted != index) & (truth == index)).sum())
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        f1.append(2 * precision * recall / max(precision + recall, 1e-12))
    severity_true = sample["severity"].cpu().numpy(); severity_pred = out["severity"].cpu().numpy()
    gm, gs = geometry_stats; tm, ts = texture_stats
    geometry_prediction = out["geometry"][sample["geometry_mask"]] * gs + gm
    texture_prediction = out["texture"][sample["texture_mask"]] * ts + tm
    geometry = vector_metrics(geometry_prediction, sample["geometry"][sample["geometry_mask"]], gm, geometry_names)
    texture = vector_metrics(texture_prediction, sample["texture"][sample["texture_mask"]], tm, texture_names)
    severity_mae = float(np.mean(np.abs(severity_pred - severity_true)))
    result = {"count": len(truth), "accuracy": float((predicted == truth).float().mean()),
              "macro_f1": float(np.mean(f1)), "severity_mae": severity_mae,
              "severity_normalized_mae": severity_mae / max(severity_baseline, 1e-12),
              "severity_srcc": correlation(rankdata(severity_pred), rankdata(severity_true)),
              "geometry": geometry, "texture": texture,
              "mean_geometry_weights": out["geometry_weights"].mean(0).cpu().tolist(),
              "mean_texture_weights": out["texture_weights"].mean(0).cpu().tolist()}
    result["selection_error"] = float(np.mean([1.0 - result["macro_f1"], result["severity_normalized_mae"],
                                                geometry["mean_normalized_mae"], texture["mean_normalized_mae"]]))
    return result


def main():
    args = parse_args(); seed_all(args.seed); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    store = FeatureStore(args.feature_dir, device)
    samples, geometry_names, texture_names = build_samples(store, args.feature_dir, args.geometry_target_dir, args.texture_target_dir, device)
    geometry_stats = stats(samples, "geometry", "geometry_mask"); texture_stats = stats(samples, "texture", "texture_mask")
    severity_mean = samples["train"]["severity"].mean(); severity_baseline = float(torch.mean(torch.abs(samples["val"]["severity"] - severity_mean)))
    model = BranchAwareQuality(store.dims, len(geometry_names), len(texture_names)).to(device)
    counts = torch.bincount(samples["train"]["attack"], minlength=len(ATTACKS)).float()
    class_weights = counts.sum() / torch.clamp(counts * len(ATTACKS), min=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best, state, history = None, None, []
    train = samples["train"]; gm, gs = geometry_stats; tm, ts = texture_stats
    for epoch in range(1, args.epochs + 1):
        model.train(); order = torch.randperm(len(train["attack"]), device=device); losses = []
        for start in range(0, len(order), 128):
            index = order[start:start + 128]; out = model([x[index] for x in train["reference"]], [x[index] for x in train["degraded"]])
            loss = F.cross_entropy(out["attack"], train["attack"][index], weight=class_weights)
            loss = loss + 0.5 * F.mse_loss(out["severity"], train["severity"][index])
            gmask = train["geometry_mask"][index]; tmask = train["texture_mask"][index]
            if gmask.any(): loss = loss + 0.5 * F.smooth_l1_loss(out["geometry"][gmask], (train["geometry"][index][gmask] - gm) / gs)
            if tmask.any(): loss = loss + 0.5 * F.smooth_l1_loss(out["texture"][tmask], (train["texture"][index][tmask] - tm) / ts)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        val = evaluate(model, samples["val"], geometry_stats, texture_stats, geometry_names, texture_names, severity_baseline)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "selection_error": val["selection_error"],
                        "macro_f1": val["macro_f1"], "geometry_nmae": val["geometry"]["mean_normalized_mae"],
                        "texture_nmae": val["texture"]["mean_normalized_mae"]})
        key = (val["selection_error"], epoch)
        if best is None or key < best: best = key; state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    results = {split: evaluate(model, samples[split], geometry_stats, texture_stats, geometry_names, texture_names, severity_baseline)
               for split in ("val", "test", "blind")}
    output = {"status": "BRANCH_AWARE_MULTITASK_QUALITY_COMPLETE", "seed": args.seed,
              "device": torch.cuda.get_device_name(device), "best_epoch": min(history, key=lambda x: (x["selection_error"], x["epoch"]))["epoch"],
              "protocol": {"base_frozen": True, "full_reference": True,
                           "selection": "minimum validation mean of classification error, normalized severity MAE, geometry NMAE, texture NMAE",
                           "test_blind_locked": True, "branch_order": ["point", "mesh", "morphology", "texture"]},
              "results": results, "history": history}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "dims": store.dims, "seed": args.seed}, args.output_dir / "quality_model.pt")
    (args.output_dir / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": output["best_epoch"], "results": {s: {"macro_f1": r["macro_f1"],
          "severity_mae": r["severity_mae"], "geometry_nmae": r["geometry"]["mean_normalized_mae"],
          "texture_nmae": r["texture"]["mean_normalized_mae"], "selection_error": r["selection_error"]}
          for s, r in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
