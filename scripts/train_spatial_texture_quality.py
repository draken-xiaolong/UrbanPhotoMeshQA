#!/usr/bin/env python3
"""Compare pooled and spatial multi-view features for no-reference texture quality."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_real_gltf_quality import regression_metrics


VARIANTS = ("pooled", "spatial", "spatial_stats")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TextureQualityModel(nn.Module):
    def __init__(self, variant: str, token_dim: int = 576, view_stat_dim: int = 12,
                 asset_stat_dim: int = 8):
        super().__init__(); self.variant = variant
        self.token_projection = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 128), nn.GELU())
        if variant == "pooled":
            self.regression = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.1),
                                            nn.Linear(128, 1), nn.Sigmoid())
            return
        self.view_embedding = nn.Parameter(torch.randn(1, 6, 1, 128) * 0.02)
        self.spatial_embedding = nn.Parameter(torch.randn(1, 1, 5, 128) * 0.02)
        if variant == "spatial_stats":
            self.view_statistics = nn.Sequential(nn.LayerNorm(view_stat_dim), nn.Linear(view_stat_dim, 128),
                                                 nn.GELU(), nn.Linear(128, 128))
            self.asset_statistics = nn.Sequential(nn.LayerNorm(asset_stat_dim), nn.Linear(asset_stat_dim, 64),
                                                  nn.GELU())
        layer = nn.TransformerEncoderLayer(128, 4, dim_feedforward=256, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.pool = nn.Linear(128, 1)
        final_dim = 192 if variant == "spatial_stats" else 128
        self.regression = nn.Sequential(nn.Linear(final_dim, 128), nn.LayerNorm(128), nn.GELU(),
                                        nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid())

    def encode(self, tokens, token_mask, view_stats, asset_stats):
        if self.variant == "pooled":
            pooled = tokens[:, :, 0].mean(dim=1)
            return self.token_projection(pooled)
        encoded = self.token_projection(tokens) + self.view_embedding + self.spatial_embedding
        if self.variant == "spatial_stats":
            encoded = encoded + self.view_statistics(view_stats)[:, :, None, :]
        batch = len(tokens); encoded = encoded.reshape(batch, 30, 128)
        mask = token_mask.reshape(batch, 30)
        encoded = self.transformer(encoded, src_key_padding_mask=~mask)
        logits = self.pool(encoded).squeeze(2).masked_fill(~mask, -1e9)
        pooled = torch.sum(torch.softmax(logits, dim=1)[:, :, None] * encoded, dim=1)
        if self.variant == "spatial_stats":
            pooled = torch.cat([pooled, self.asset_statistics(asset_stats)], dim=1)
        return pooled

    def forward(self, tokens, token_mask, view_stats, asset_stats):
        return self.regression(self.encode(tokens, token_mask, view_stats, asset_stats)).squeeze(1)


def load_data(root: Path, split: str, device: torch.device, statistics=None):
    with np.load(root / f"features_{split}.npz") as values:
        raw = {name: values[name].copy() for name in values.files}
    if statistics is None:
        statistics = {}
        for name in ("view_stats", "asset_stats"):
            value = raw[name].astype(np.float32)
            axes = (0, 1) if name == "view_stats" else 0
            statistics[name] = {"mean": value.mean(axis=axes),
                                "std": np.maximum(value.std(axis=axes), 1e-5)}
    normalized = {}
    for name in ("view_stats", "asset_stats"):
        normalized[name] = ((raw[name] - statistics[name]["mean"]) /
                            statistics[name]["std"]).astype(np.float32)
    data = {
        "tokens": torch.from_numpy(raw["tokens"].astype(np.float32)).to(device),
        "token_mask": torch.from_numpy(raw["token_mask"]).to(device),
        "view_stats": torch.from_numpy(normalized["view_stats"]).to(device),
        "asset_stats": torch.from_numpy(normalized["asset_stats"]).to(device),
        "target": torch.from_numpy(raw["texture_quality"].astype(np.float32)).to(device),
        "asset_ids": raw["asset_ids"].astype(str), "attacks": raw["attacks"].astype(str),
        "levels": raw["levels"].astype(str),
    }
    return data, statistics


@torch.no_grad()
def evaluate(model, data):
    model.eval(); prediction = []
    for start in range(0, len(data["target"]), 128):
        index = slice(start, start + 128)
        prediction.append(model(data["tokens"][index], data["token_mask"][index],
                                data["view_stats"][index], data["asset_stats"][index]).cpu().numpy())
    prediction = np.concatenate(prediction); target = data["target"].cpu().numpy()
    result = {"overall": regression_metrics(prediction, target), "per_attack": {}, "per_level": {}}
    for attack in sorted(set(data["attacks"])):
        mask = data["attacks"] == attack
        result["per_attack"][attack] = regression_metrics(prediction[mask], target[mask])
    for level in ("light", "medium", "heavy"):
        mask = data["levels"] == level
        result["per_level"][level] = regression_metrics(prediction[mask], target[mask])
    monotonic = total = 0
    for asset_id in sorted(set(data["asset_ids"])):
        for attack in ("texture_detail_loss", "texture_region_missing", "texture_misalignment"):
            indices = [np.flatnonzero((data["asset_ids"] == asset_id) & (data["attacks"] == attack)
                                     & (data["levels"] == level)) for level in ("light", "medium", "heavy")]
            if all(len(value) == 1 for value in indices):
                qualities = [prediction[value[0]] for value in indices]
                monotonic += int(qualities[0] >= qualities[1] >= qualities[2]); total += 1
    result["severity_monotonic"] = {"passed": monotonic, "total": total,
                                     "rate": monotonic / max(total, 1)}
    return result


def train_variant(variant, train, val, args, device):
    seed_all(args.seed); model = TextureQualityModel(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    best_key = best_state = best_epoch = None; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = torch.randperm(len(train["target"]), generator=generator, device=device)
        losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            prediction = model(train["tokens"][index], train["token_mask"][index],
                               train["view_stats"][index], train["asset_stats"][index])
            target = train["target"][index]
            loss = 3.0 * F.smooth_l1_loss(prediction, target)
            delta = target[:, None] - target[None, :]
            valid = torch.abs(delta) > 0.03
            if valid.any():
                prediction_delta = prediction[:, None] - prediction[None, :]
                loss = loss + 0.2 * F.softplus(
                    -5.0 * torch.sign(delta[valid]) * prediction_delta[valid]).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, val)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": metrics})
        key = (metrics["overall"]["srcc"], -metrics["overall"]["mae"])
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{variant} epoch={epoch} loss={np.mean(losses):.4f} "
                  f"val_mae={metrics['overall']['mae']:.4f} val_srcc={metrics['overall']['srcc']:.4f}", flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state); metrics = evaluate(model, val)
    return model, {"variant": variant, "parameters": sum(p.numel() for p in model.parameters()),
                   "best_epoch": best_epoch, "epochs_run": len(history), "val": metrics,
                   "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    train, statistics = load_data(args.feature_dir, "train", device)
    val, _ = load_data(args.feature_dir, "val", device, statistics)
    args.output_dir.mkdir(parents=True, exist_ok=True); results = {}
    serializable_statistics = {name: {key: value.tolist() for key, value in item.items()}
                               for name, item in statistics.items()}
    for variant in args.variants:
        model, result = train_variant(variant, train, val, args, device)
        torch.save({"schema_version": 1, "seed": args.seed, "variant": variant,
                    "model": model.state_dict(), "statistics": serializable_statistics,
                    "protocol": {"loaded_splits": ["train", "val"], "test_blind_loaded": False}},
                   args.output_dir / f"{variant}.pt")
        results[variant] = result
    selected = max(results, key=lambda name: (results[name]["val"]["overall"]["srcc"],
                                               -results[name]["val"]["overall"]["mae"]))
    payload = {"schema_version": 1, "status": "COMPLETE", "seed": args.seed,
               "protocol": {"selection": "Val Texture SRCC then MAE", "loaded_splits": ["train", "val"],
                            "test_blind_loaded": False}, "selected": selected, "variants": results}
    (args.output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "val": {name: result["val"]["overall"]
                                                      for name, result in results.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
