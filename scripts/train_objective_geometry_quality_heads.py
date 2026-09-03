#!/usr/bin/env python3
"""Compare frozen representations for full-reference objective geometry-quality prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_four_branch_fusion import ConcatMLP, FeatureStore
from train_frozen_base_quality_head import correlation, rankdata, seed_all


class Comparator(nn.Module):
    def __init__(self, dim: int, outputs: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim * 2 + 1), nn.Linear(dim * 2 + 1, 512),
                                 nn.GELU(), nn.Dropout(0.15), nn.Linear(512, 256),
                                 nn.GELU(), nn.Linear(256, outputs))

    def forward(self, reference, degraded):
        cosine = F.cosine_similarity(reference, degraded, dim=1, eps=1e-6).unsqueeze(1)
        return self.net(torch.cat([torch.abs(reference - degraded), reference * degraded, cosine], 1))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-name", default="objective_geometry_quality")
    return parser.parse_args()


@torch.no_grad()
def build_samples(base, store, feature_dir: Path, target_dir: Path, mode: str):
    output, metric_names = {}, None
    for split in ("train", "val", "test", "blind"):
        data = store.data[split]
        gallery_raw = torch.cat(data["gallery"], 1); query_raw = torch.cat(data["query"], 1)
        gallery_identity, _ = base(data["gallery"]); query_identity, _ = base(data["query"])
        if mode == "identity": gallery, query = gallery_identity, query_identity
        elif mode == "sidepath": gallery, query = gallery_raw, query_raw
        elif mode == "identity_plus_sidepath":
            gallery = torch.cat([gallery_identity, gallery_raw], 1)
            query = torch.cat([query_identity, query_raw], 1)
        else: raise ValueError(mode)
        with np.load(feature_dir / f"scores_{split}.npz") as features, np.load(target_dir / f"targets_{split}.npz") as targets:
            index = targets["query_indices"].astype(np.int64)
            gallery_index = features["targets"][index].astype(np.int64)
            y = targets["targets"].astype(np.float32)
            metric_names = targets["metrics"].astype(str).tolist()
        output[split] = (gallery[torch.as_tensor(gallery_index, device=gallery.device)],
                         query[torch.as_tensor(index, device=query.device)],
                         torch.as_tensor(y, device=query.device))
    return output, metric_names


@torch.no_grad()
def evaluate(model, sample, mean, std, names):
    model.eval(); reference, degraded, target = sample
    prediction = model(reference, degraded) * std + mean
    pred = prediction.cpu().numpy(); truth = target.cpu().numpy(); center = mean.cpu().numpy()
    metrics = {}
    for i, name in enumerate(names):
        mae = float(np.mean(np.abs(pred[:, i] - truth[:, i])))
        baseline = float(np.mean(np.abs(truth[:, i] - center[i])))
        metrics[name] = {"mae": mae, "mean_baseline_mae": baseline,
                         "normalized_mae": mae / max(baseline, 1e-12),
                         "plcc": correlation(pred[:, i], truth[:, i]),
                         "srcc": correlation(rankdata(pred[:, i]), rankdata(truth[:, i]))}
    return {"count": len(truth), "mean_normalized_mae": float(np.mean([v["normalized_mae"] for v in metrics.values()])),
            "metrics": metrics}


def train_model(model, samples, names, args):
    train = samples["train"]; mean = train[2].mean(0); std = train[2].std(0).clamp_min(1e-8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best, state, history = None, None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = torch.randperm(len(train[2]), device=train[2].device); losses = []
        for start in range(0, len(order), 128):
            idx = order[start:start + 128]
            prediction = model(train[0][idx], train[1][idx])
            loss = F.smooth_l1_loss(prediction, (train[2][idx] - mean) / std)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        val = evaluate(model, samples["val"], mean, std, names)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "validation_mean_normalized_mae": val["mean_normalized_mae"]})
        key = (val["mean_normalized_mae"], epoch)
        if best is None or key < best:
            best = key; state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, mean, std, history


def main():
    args = parse_args(); seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    store = FeatureStore(args.feature_dir, device)
    checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    base = ConcatMLP(checkpoint["dims"]).to(device).eval(); base.load_state_dict(checkpoint["model"])
    for p in base.parameters(): p.requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.target_dir / "metadata.json"
    target_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    output = {"status": "OBJECTIVE_QUALITY_COMPARISON_COMPLETE", "seed": args.seed,
              "device": torch.cuda.get_device_name(device),
              "protocol": {"task": args.task_name, "target_metadata": target_metadata,
                           "selection": "lowest validation mean normalized MAE; test/blind locked",
                           "base_frozen": True}, "models": {}}
    for mode in ("identity", "sidepath", "identity_plus_sidepath"):
        seed_all(args.seed); samples, names = build_samples(base, store, args.feature_dir, args.target_dir, mode)
        dim = samples["train"][0].shape[1]
        model, mean, std, history = train_model(Comparator(dim, len(names)).to(device), samples, names, args)
        results = {split: evaluate(model, samples[split], mean, std, names) for split in ("val", "test", "blind")}
        best_epoch = min(history, key=lambda row: (row["validation_mean_normalized_mae"], row["epoch"]))["epoch"]
        output["models"][mode] = {"input_dim": dim, "best_epoch": best_epoch,
                                  "results": results, "history": history}
        torch.save({"model": model.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
                    "input_dim": dim, "metrics": names, "seed": args.seed}, args.output_dir / f"{mode}.pt")
    (args.output_dir / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {split: values["results"][split]["mean_normalized_mae"]
                            for split in ("val", "test", "blind")}
                      for name, values in output["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
