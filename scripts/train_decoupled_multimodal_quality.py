#!/usr/bin/env python3
"""Compare shared versus modality-decoupled global quality heads on Train/Val."""

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
from train_spatial_texture_quality import TextureQualityModel


VARIANTS = ("shared", "decoupled", "decoupled_fusion")
BRANCH_DIMS = {"point_global": 521, "mesh_global": 1032, "morphology": 13}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


class GeometryTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleDict({name: nn.Sequential(
            nn.LayerNorm(size), nn.Linear(size, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
            for name, size in BRANCH_DIMS.items()})
        self.patch_projection = nn.Sequential(nn.LayerNorm(58), nn.Linear(58, 128), nn.GELU(),
                                              nn.Linear(128, 128))
        self.patch_pool = nn.Linear(128, 1)
        self.embedding = nn.Parameter(torch.randn(1, 4, 128) * 0.02)
        self.attention = nn.MultiheadAttention(128, 4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.output = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
                                    nn.Dropout(0.1), nn.Linear(256, 128), nn.GELU())

    def forward(self, data, index):
        tokens = [self.projections[name](data[name][index]) for name in BRANCH_DIMS]
        patches = self.patch_projection(data["patches"][index])
        patch_mask = data["patch_mask"][index]
        logits = self.patch_pool(patches).squeeze(2).masked_fill(~patch_mask, -1e9)
        tokens.append(torch.sum(torch.softmax(logits, 1)[:, :, None] * patches, 1))
        tokens = torch.stack(tokens, 1) + self.embedding
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return self.output(self.norm(tokens + attended).flatten(1))


class MultimodalQualityModel(nn.Module):
    def __init__(self, variant, texture_checkpoint):
        super().__init__(); self.variant = variant; self.geometry = GeometryTrunk()
        self.texture = TextureQualityModel("spatial_stats")
        if texture_checkpoint is not None:
            self.texture.load_state_dict(texture_checkpoint["model"])
        if variant == "shared":
            self.shared = nn.Sequential(nn.Linear(128 + 192, 256), nn.LayerNorm(256), nn.GELU(),
                                        nn.Dropout(0.1), nn.Linear(256, 128), nn.GELU(),
                                        nn.Linear(128, 3), nn.Sigmoid())
        else:
            self.geometry_quality = nn.Sequential(nn.Linear(128, 64), nn.GELU(),
                                                  nn.Linear(64, 1), nn.Sigmoid())
            if variant == "decoupled_fusion":
                self.overall_fusion = nn.Sequential(
                    nn.Linear(128 + 192 + 2, 192), nn.LayerNorm(192), nn.GELU(),
                    nn.Dropout(0.1), nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, data, index):
        geometry_latent = self.geometry(data, index)
        texture_latent = self.texture.encode(data["tokens"][index], data["token_mask"][index],
                                             data["view_stats"][index], data["asset_stats"][index])
        if self.variant == "shared":
            output = self.shared(torch.cat([geometry_latent, texture_latent], dim=1))
            return {"overall": output[:, 0], "geometry": output[:, 1], "texture": output[:, 2]}
        geometry_quality = self.geometry_quality(geometry_latent).squeeze(1)
        texture_quality = self.texture.regression(texture_latent).squeeze(1)
        if self.variant == "decoupled_fusion":
            overall = self.overall_fusion(torch.cat([
                geometry_latent.detach(), texture_latent.detach(), geometry_quality.detach()[:, None],
                texture_quality.detach()[:, None]
            ], dim=1)).squeeze(1)
        else:
            overall = torch.minimum(geometry_quality, texture_quality)
        return {"overall": overall, "geometry": geometry_quality, "texture": texture_quality}


def load_data(root, split, device, statistics=None, fixed_statistics=None):
    with np.load(root / f"features_{split}.npz") as values:
        raw = {name: values[name].copy() for name in values.files}
    statistic_names = (*BRANCH_DIMS, "patches", "view_stats", "asset_stats")
    if statistics is None:
        statistics = {}
        for name in statistic_names:
            if fixed_statistics and name in fixed_statistics:
                statistics[name] = fixed_statistics[name]
            else:
                value = raw[name].astype(np.float32)
                axes = tuple(range(value.ndim - 1))
                statistics[name] = {"mean": value.mean(axis=axes),
                                    "std": np.maximum(value.std(axis=axes), 1e-5)}
    data = {}
    for name in statistic_names:
        data[name] = torch.from_numpy(((raw[name] - statistics[name]["mean"])
                                       / statistics[name]["std"]).astype(np.float32)).to(device)
    data.update({
        "tokens": torch.from_numpy(raw["tokens"].astype(np.float32)).to(device),
        "token_mask": torch.from_numpy(raw["token_mask"]).to(device),
        "patch_mask": torch.from_numpy(raw["patch_mask"]).to(device),
        "overall": torch.from_numpy(raw["overall_quality"].astype(np.float32)).to(device),
        "geometry_target": torch.from_numpy(raw["geometry_quality"].astype(np.float32)).to(device),
        "texture_target": torch.from_numpy(raw["texture_quality"].astype(np.float32)).to(device),
        "asset_ids": raw["asset_ids"].astype(str), "attacks": raw["attacks"].astype(str),
        "levels": raw["levels"].astype(str), "sheets": raw["sheets"].astype(str),
    })
    asset_names = sorted(set(raw["asset_ids"].astype(str)))
    asset_lookup = {name: index for index, name in enumerate(asset_names)}
    data["asset_group"] = torch.tensor(
        [asset_lookup[name] for name in raw["asset_ids"].astype(str)], dtype=torch.long, device=device)
    return data, statistics


@torch.no_grad()
def evaluate(model, data):
    model.eval(); collected = {name: [] for name in ("overall", "geometry", "texture")}
    for start in range(0, len(data["overall"]), 128):
        output = model(data, slice(start, start + 128))
        for name in collected: collected[name].append(output[name].cpu().numpy())
    prediction = {name: np.concatenate(value) for name, value in collected.items()}
    target = {"overall": data["overall"].cpu().numpy(),
              "geometry": data["geometry_target"].cpu().numpy(),
              "texture": data["texture_target"].cpu().numpy()}
    result = {name: regression_metrics(prediction[name], target[name]) for name in prediction}
    result["per_attack"] = {}
    for attack in sorted(set(data["attacks"])):
        mask = data["attacks"] == attack
        result["per_attack"][attack] = regression_metrics(prediction["overall"][mask], target["overall"][mask])
    result["per_level"] = {}
    for level in ("light", "medium", "heavy"):
        mask = data["levels"] == level
        result["per_level"][level] = regression_metrics(prediction["overall"][mask], target["overall"][mask])
    return result


def train_variant(variant, train, val, texture_checkpoint, args, device):
    seed_all(args.seed); model = MultimodalQualityModel(variant, texture_checkpoint).to(device)
    texture_parameters = list(model.texture.parameters())
    texture_ids = {id(parameter) for parameter in texture_parameters}
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in texture_ids]
    optimizer = torch.optim.AdamW([{"params": other_parameters, "lr": args.lr},
                                   {"params": texture_parameters, "lr": args.texture_lr}], weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    best_key = best_state = best_epoch = None; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = torch.randperm(len(train["overall"]), device=device, generator=generator)
        if args.grouped_ranking:
            group_order = torch.randperm(int(train["asset_group"].max()) + 1,
                                         device=device, generator=generator)
            batches = []
            for start in range(0, len(group_order), args.batch_buildings):
                selected_groups = group_order[start:start + args.batch_buildings]
                mask = (train["asset_group"][:, None] == selected_groups[None, :]).any(dim=1)
                index = torch.nonzero(mask, as_tuple=False).flatten()
                batches.append(index[torch.randperm(len(index), device=device, generator=generator)])
        else:
            batches = [order[start:start + args.batch_size]
                       for start in range(0, len(order), args.batch_size)]
        losses = []
        for index in batches:
            output = model(train, index)
            geometry_loss = F.smooth_l1_loss(output["geometry"], train["geometry_target"][index])
            texture_loss = F.smooth_l1_loss(output["texture"], train["texture_target"][index])
            if args.modality_loss_reweighting:
                names = train["attacks"][index.cpu().numpy()]
                geometry_weight = torch.from_numpy((~np.isin(names, [
                    "texture_detail_loss", "texture_region_missing", "texture_misalignment"
                ])).astype(np.float32)).to(device)
                texture_weight = torch.from_numpy(np.where(np.isin(names, [
                    "geometry_hole", "mesh_simplification_qem", "geometry_noise_spike"
                ]), 0.25, 1.0).astype(np.float32)).to(device)
                geometry_loss = (F.smooth_l1_loss(
                    output["geometry"], train["geometry_target"][index], reduction="none")
                    * geometry_weight).sum() / geometry_weight.sum().clamp_min(1)
                texture_loss = (F.smooth_l1_loss(
                    output["texture"], train["texture_target"][index], reduction="none")
                    * texture_weight).sum() / texture_weight.sum().clamp_min(1)
            loss = (3.0 * F.smooth_l1_loss(output["overall"], train["overall"][index])
                    + 1.5 * geometry_loss + 1.5 * texture_loss)
            delta = train["overall"][index][:, None] - train["overall"][index][None, :]
            valid = torch.abs(delta) > 0.03
            if args.grouped_ranking:
                groups = train["asset_group"][index]
                valid = valid & (groups[:, None] == groups[None, :])
            if valid.any():
                prediction_delta = output["overall"][:, None] - output["overall"][None, :]
                loss = loss + 0.2 * F.softplus(
                    -5.0 * torch.sign(delta[valid]) * prediction_delta[valid]).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach()))
        metrics = evaluate(model, val); history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": metrics})
        key = (metrics["overall"]["srcc"], min(metrics["geometry"]["srcc"], metrics["texture"]["srcc"]),
               -metrics["overall"]["mae"])
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else: stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{variant} epoch={epoch} loss={np.mean(losses):.4f} oqi={metrics['overall']['srcc']:.4f} "
                  f"g={metrics['geometry']['srcc']:.4f} t={metrics['texture']['srcc']:.4f}", flush=True)
        if stale >= args.patience: break
    model.load_state_dict(best_state); metrics = evaluate(model, val)
    return model, {"variant": variant, "best_epoch": best_epoch, "epochs_run": len(history),
                   "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "val": metrics, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--texture-checkpoint", type=Path,
                        help="Optional Train-only spatial texture initialization")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--epochs", type=int, default=120); parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--texture-lr", type=float, default=1e-4); parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grouped-ranking", action="store_true",
                        help="Batch complete same-building attack groups and rank only within buildings")
    parser.add_argument("--batch-buildings", type=int, default=4)
    parser.add_argument("--modality-loss-reweighting", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_all(args.seed); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    texture_checkpoint = (torch.load(args.texture_checkpoint, map_location=device, weights_only=False)
                          if args.texture_checkpoint else None)
    fixed_statistics = (None if texture_checkpoint is None else {
        name: {key: np.asarray(value, np.float32) for key, value in item.items()}
        for name, item in texture_checkpoint["statistics"].items()})
    train, statistics = load_data(args.feature_dir, "train", device, fixed_statistics=fixed_statistics)
    val, _ = load_data(args.feature_dir, "val", device, statistics)
    args.output_dir.mkdir(parents=True, exist_ok=True); results = {}
    serialized = {name: {key: value.tolist() for key, value in item.items()} for name, item in statistics.items()}
    for variant in args.variants:
        model, result = train_variant(variant, train, val, texture_checkpoint, args, device)
        torch.save({"schema_version": 1, "seed": args.seed, "variant": variant, "model": model.state_dict(),
                    "statistics": serialized, "protocol": {"loaded_splits": ["train", "val"],
                                                             "test_blind_loaded": False,
                                                             "grouped_ranking": args.grouped_ranking,
                                                             "batch_buildings": args.batch_buildings,
                                                             "modality_loss_reweighting": args.modality_loss_reweighting}},
                   args.output_dir / f"{variant}.pt"); results[variant] = result
    selected = max(results, key=lambda name: (results[name]["val"]["overall"]["srcc"],
                                               min(results[name]["val"]["geometry"]["srcc"],
                                                   results[name]["val"]["texture"]["srcc"]),
                                               -results[name]["val"]["overall"]["mae"]))
    payload = {"schema_version": 1, "status": "COMPLETE", "seed": args.seed,
               "protocol": {"loaded_splits": ["train", "val"], "test_blind_loaded": False,
                            "selection": "Val OQI SRCC, min component SRCC, OQI MAE",
                            "grouped_ranking": args.grouped_ranking,
                            "batch_buildings": args.batch_buildings,
                            "modality_loss_reweighting": args.modality_loss_reweighting},
               "selected": selected, "variants": results}
    (args.output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "val": {name: value["val"] for name, value in results.items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
