#!/usr/bin/env python3
"""Train a degradation diagnosis head on the frozen four-branch identity Base."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_four_branch_fusion import ConcatMLP, FeatureStore


ATTACKS = (
    "clean", "background_shift", "blur1.5", "brightness0.55", "camera_jitter12",
    "connected_crop", "downsample32", "hole", "jpeg50", "occlusion20", "retriangulate",
)
ATTACK_TO_INDEX = {name: index for index, name in enumerate(ATTACKS)}
FIXED_SEVERITY = {
    "clean": 0.0, "background_shift": 0.5, "blur1.5": 0.5, "brightness0.55": 0.45,
    "camera_jitter12": 0.5, "downsample32": 2.0 / 3.0, "jpeg50": 0.5,
    "occlusion20": 0.2,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class QualityHead(nn.Module):
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1))
        self.attack = nn.Linear(128, len(ATTACKS))
        self.severity = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x):
        hidden = self.shared(x)
        return self.attack(hidden), self.severity(hidden).squeeze(1)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def correlation(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def make_labels(attacks: np.ndarray, severities: np.ndarray):
    classes, quality = [], []
    for attack, severity in zip(attacks.tolist(), severities.tolist()):
        classes.append(ATTACK_TO_INDEX[attack])
        quality.append(FIXED_SEVERITY.get(attack, float(severity)))
    return np.asarray(classes, np.int64), np.asarray(quality, np.float32)


@torch.no_grad()
def frozen_embeddings(base, store, root: Path, split: str):
    data = store.data[split]
    gallery, _ = base(data["gallery"])
    query, _ = base(data["query"])
    with np.load(root / f"scores_{split}.npz") as values:
        attacks = values["attacks"].astype(str)
        severities = values["severities"].astype(np.float32)
    x = torch.cat([gallery, query], dim=0)
    names = np.concatenate([np.full(len(gallery), "clean"), attacks])
    raw_severity = np.concatenate([np.zeros(len(gallery), np.float32), severities])
    classes, severity = make_labels(names, raw_severity)
    return x.detach(), torch.as_tensor(classes, device=x.device), torch.as_tensor(severity, device=x.device)


@torch.no_grad()
def evaluate(head, sample):
    head.eval(); x, labels, severity = sample
    logits, predicted_severity = head(x)
    predicted = logits.argmax(1)
    per_class = []
    for index, name in enumerate(ATTACKS):
        tp = int(((predicted == index) & (labels == index)).sum())
        fp = int(((predicted == index) & (labels != index)).sum())
        fn = int(((predicted != index) & (labels == index)).sum())
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class.append({"class": name, "count": int((labels == index).sum()), "f1": f1})
    truth = severity.cpu().numpy(); estimate = predicted_severity.cpu().numpy()
    return {
        "count": len(labels), "accuracy": float((predicted == labels).float().mean()),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "per_class": per_class, "severity_mae": float(np.mean(np.abs(estimate - truth))),
        "severity_plcc": correlation(estimate, truth),
        "severity_srcc": correlation(rankdata(estimate), rankdata(truth)),
    }


def main():
    args = parse_args(); seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal quality-head pilot must run on CUDA")
    store = FeatureStore(args.feature_dir, device)
    checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    base = ConcatMLP(checkpoint["dims"]).to(device).eval()
    base.load_state_dict(checkpoint["model"])
    for parameter in base.parameters(): parameter.requires_grad_(False)
    samples = {split: frozen_embeddings(base, store, args.feature_dir, split)
               for split in ("train", "val", "test", "blind")}
    head = QualityHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, history = None, None, []
    x, labels, severity = samples["train"]
    class_count = torch.bincount(labels, minlength=len(ATTACKS)).float()
    class_weights = class_count.sum() / torch.clamp(class_count * len(ATTACKS), min=1.0)
    for epoch in range(1, args.epochs + 1):
        head.train(); order = torch.randperm(len(x), device=device); losses = []
        for start in range(0, len(order), 128):
            index = order[start:start + 128]
            logits, prediction = head(x[index])
            loss = F.cross_entropy(logits, labels[index], weight=class_weights) + 0.5 * F.mse_loss(prediction, severity[index])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        val = evaluate(head, samples["val"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "macro_f1": val["macro_f1"], "severity_mae": val["severity_mae"]})
        key = (val["macro_f1"], -val["severity_mae"], -epoch)
        if best is None or key > best:
            best, state = key, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(state)
    results = {split: evaluate(head, sample) for split, sample in samples.items() if split != "train"}
    majority = max(int((samples["train"][1] == i).sum()) for i in range(len(ATTACKS))) / len(samples["train"][1])
    mean_severity = float(samples["train"][2].mean())
    mean_baseline_mae = {
        split: float(torch.mean(torch.abs(sample[2] - mean_severity)))
        for split, sample in samples.items() if split != "train"
    }
    output = {
        "status": "FROZEN_IDENTITY_BASE_QUALITY_PILOT_COMPLETE", "seed": args.seed,
        "device": torch.cuda.get_device_name(device), "best_epoch": max(history, key=lambda v: (v["macro_f1"], -v["severity_mae"], -v["epoch"]))["epoch"],
        "protocol": {"base": "frozen four-branch concat MLP 256D identity feature",
                     "task": "11-class degradation diagnosis plus normalized attack-parameter regression",
                     "selection": "validation macro-F1 then severity MAE; test/blind locked",
                     "scope": "feasibility pilot, not subjective perceptual quality or objective full-reference quality"},
        "baselines": {"majority_accuracy": majority, "train_mean_severity": mean_severity,
                      "mean_severity_mae": mean_baseline_mae},
        "results": results, "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": head.state_dict(), "attacks": ATTACKS, "seed": args.seed}, args.output_dir / "quality_head.pt")
    (args.output_dir / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": output["best_epoch"], "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
