#!/usr/bin/env python3
"""Compare score fusion, concatenation MLP, and learned confidence-gated fusion."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BRANCHES = ("point", "mesh", "morphology", "texture")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def metric(scores: np.ndarray, targets: np.ndarray) -> dict:
    order = np.argsort(-scores, axis=1)
    ranks = np.asarray([np.flatnonzero(row == target)[0] + 1 for row, target in zip(order, targets)])
    return {"count": int(len(ranks)), "r1": float(np.mean(ranks == 1)),
            "r5": float(np.mean(ranks <= 5)), "mrr": float(np.mean(1.0 / ranks))}


class FeatureStore:
    def __init__(self, root: Path, device: torch.device):
        raw = {}
        for split in ("train", "val", "test", "blind"):
            with np.load(root / f"scores_{split}.npz") as z:
                raw[split] = {k: z[k].copy() for k in z.files}
        stats = {}
        for branch in BRANCHES:
            values = raw["train"][f"gallery_{branch}"].astype(np.float32)
            stats[branch] = (values.mean(0, keepdims=True), np.maximum(values.std(0, keepdims=True), 1e-5))
        self.data = {}
        for split, values in raw.items():
            item = {"targets": torch.as_tensor(values["targets"], dtype=torch.long, device=device)}
            for side in ("gallery", "query"):
                tensors = []
                for branch in BRANCHES:
                    x = values[f"{side}_{branch}"].astype(np.float32)
                    mean, std = stats[branch]
                    x = (x - mean) / std
                    tensors.append(torch.as_tensor(x, device=device))
                item[side] = tensors
            self.data[split] = item
        self.dims = [self.data["train"]["gallery"][i].shape[1] for i in range(4)]


class ConcatMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sum(dims), 512), nn.LayerNorm(512), nn.GELU(),
                                 nn.Dropout(0.10), nn.Linear(512, 256))

    def forward(self, xs, training_aug=False):
        return F.normalize(self.net(torch.cat(xs, dim=1)), dim=1), None


class GatedFusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 256), nn.LayerNorm(256), nn.GELU()) for dim in dims
        ])
        self.gate = nn.Sequential(nn.Linear(256 * 4, 256), nn.GELU(), nn.Linear(256, 4))

    def forward(self, xs, training_aug=False):
        branches = torch.stack([F.normalize(layer(x), dim=1) for layer, x in zip(self.projections, xs)], 1)
        logits = self.gate(branches.flatten(1))
        if training_aug:
            # Randomly remove texture evidence so geometry remains usable when texture is absent/corrupted.
            drop = torch.rand(len(logits), device=logits.device) < 0.25
            logits[drop, 3] = -20.0
        weights = torch.softmax(logits, dim=1)
        fused = F.normalize(torch.sum(weights[:, :, None] * branches, dim=1), dim=1)
        return fused, weights


@torch.no_grad()
def evaluate(model, split: dict) -> tuple[dict, list[float] | None]:
    model.eval()
    gallery, gallery_gate = model(split["gallery"])
    query, query_gate = model(split["query"])
    scores = (query @ gallery.T).cpu().numpy()
    gates = None if query_gate is None else query_gate.mean(0).cpu().tolist()
    return metric(scores, split["targets"].cpu().numpy()), gates


def train_model(model, store, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    train = store.data["train"]
    best, best_state, history = None, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train["targets"]), device=train["targets"].device)
        losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            gallery, _ = model(train["gallery"], training_aug=True)
            query, _ = model([x[index] for x in train["query"]], training_aug=True)
            loss = F.cross_entropy((query @ gallery.T) / 0.07, train["targets"][index])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        val, _ = evaluate(model, store.data["val"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **val})
        key = (val["r1"], val["r5"], val["mrr"], -epoch)
        if best is None or key > best:
            best = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def main() -> None:
    args = parse_args(); seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal fusion training must run on CUDA")
    store = FeatureStore(args.feature_dir, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline_results.read_text(encoding="utf-8"))
    output = {
        "status": "FOUR_BRANCH_FUSION_COMPARISON_COMPLETE", "seed": args.seed,
        "device": torch.cuda.get_device_name(device),
        "protocol": {
            "train": "80 identities from five train sheets; 12 attacked queries per identity",
            "selection": "best epoch by validation R@1/R@5/MRR; test and blind locked",
            "encoders": "four frozen encoders; only fusion heads trained",
            "texture_dropout": 0.25,
        },
        "models": {
            "score_fusion": {split: baseline["splits"][split]["stage_4_pmmt"]["overall"]
                             for split in ("val", "test", "blind")}
        },
    }
    for name, model_class in (("concat_mlp", ConcatMLP),
                              ("confidence_gated", GatedFusion)):
        seed_all(args.seed)
        model = model_class(store.dims).to(device)
        model, history = train_model(model, store, args)
        results, gates = {}, {}
        for split in ("val", "test", "blind"):
            results[split], gates[split] = evaluate(model, store.data[split])
        best_epoch = max(history, key=lambda x: (x["r1"], x["r5"], x["mrr"], -x["epoch"]))["epoch"]
        output["models"][name] = {"best_epoch": best_epoch, "metrics": results,
                                  "mean_query_gates": gates, "history": history}
        torch.save({"model": model.state_dict(), "dims": store.dims, "seed": args.seed},
                   args.output_dir / f"{name}.pt")
    path = args.output_dir / "results.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value if name == "score_fusion" else value["metrics"]
                      for name, value in output["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
