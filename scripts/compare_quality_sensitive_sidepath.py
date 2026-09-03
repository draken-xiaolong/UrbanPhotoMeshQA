#!/usr/bin/env python3
"""Compare frozen identity output with pre-fusion quality-sensitive side paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_four_branch_fusion import ConcatMLP, FeatureStore
from train_frozen_base_quality_head import (
    ATTACKS, QualityHead, evaluate, make_labels, seed_all,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--identity-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def build_samples(base, store, root: Path, mode: str):
    output = {}
    for split in ("train", "val", "test", "blind"):
        data = store.data[split]
        gallery_raw = torch.cat(data["gallery"], dim=1)
        query_raw = torch.cat(data["query"], dim=1)
        if mode == "sidepath":
            gallery, query = gallery_raw, query_raw
        elif mode == "identity_plus_sidepath":
            gallery_identity, _ = base(data["gallery"])
            query_identity, _ = base(data["query"])
            gallery = torch.cat([gallery_identity, gallery_raw], dim=1)
            query = torch.cat([query_identity, query_raw], dim=1)
        else:
            raise ValueError(mode)
        with np.load(root / f"scores_{split}.npz") as values:
            attacks = values["attacks"].astype(str)
            severities = values["severities"].astype(np.float32)
        names = np.concatenate([np.full(len(gallery), "clean"), attacks])
        raw_severity = np.concatenate([np.zeros(len(gallery), np.float32), severities])
        labels, severity = make_labels(names, raw_severity)
        output[split] = (
            torch.cat([gallery, query]).detach(),
            torch.as_tensor(labels, device=gallery.device),
            torch.as_tensor(severity, device=gallery.device),
        )
    return output


def train(head, samples, args):
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    x, labels, severity = samples["train"]
    counts = torch.bincount(labels, minlength=len(ATTACKS)).float()
    weights = counts.sum() / torch.clamp(counts * len(ATTACKS), min=1.0)
    best, state, history = None, None, []
    for epoch in range(1, args.epochs + 1):
        head.train(); order = torch.randperm(len(x), device=x.device); losses = []
        for start in range(0, len(order), 128):
            index = order[start:start + 128]
            logits, prediction = head(x[index])
            loss = F.cross_entropy(logits, labels[index], weight=weights) + 0.5 * F.mse_loss(prediction, severity[index])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        val = evaluate(head, samples["val"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "macro_f1": val["macro_f1"], "severity_mae": val["severity_mae"]})
        key = (val["macro_f1"], -val["severity_mae"], -epoch)
        if best is None or key > best:
            best = key; state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(state)
    return head, history


def main():
    args = parse_args(); seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal side-path comparison must run on CUDA")
    store = FeatureStore(args.feature_dir, device)
    checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    base = ConcatMLP(checkpoint["dims"]).to(device).eval(); base.load_state_dict(checkpoint["model"])
    for parameter in base.parameters(): parameter.requires_grad_(False)
    identity = json.loads(args.identity_results.read_text(encoding="utf-8"))
    output = {
        "status": "QUALITY_SENSITIVE_SIDEPATH_COMPARISON_COMPLETE", "seed": args.seed,
        "device": torch.cuda.get_device_name(device),
        "protocol": {"base_frozen": True, "selection": "validation macro-F1 then severity MAE",
                     "test_blind_locked": True, "targets": "attack class and normalized parameter severity"},
        "models": {"identity_256": {"results": identity["results"], "best_epoch": identity["best_epoch"]}},
    }
    for mode in ("sidepath", "identity_plus_sidepath"):
        seed_all(args.seed)
        samples = build_samples(base, store, args.feature_dir, mode)
        input_dim = samples["train"][0].shape[1]
        head, history = train(QualityHead(input_dim).to(device), samples, args)
        results = {split: evaluate(head, samples[split]) for split in ("val", "test", "blind")}
        best_epoch = max(history, key=lambda row: (row["macro_f1"], -row["severity_mae"], -row["epoch"]))["epoch"]
        output["models"][mode] = {"input_dim": input_dim, "best_epoch": best_epoch,
                                  "results": results, "history": history}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": head.state_dict(), "input_dim": input_dim, "seed": args.seed},
                   args.output_dir / f"{mode}.pt")
    path = args.output_dir / "results.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value["results"] for name, value in output["models"].items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
