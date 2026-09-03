#!/usr/bin/env python3
"""Validate partial Point/Mesh Base unfreezing on full multimodal Train/Val data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from urbanphotomeshqa.data import pad_mesh_graphs
from urbanphotomeshqa.integrity import sha256_file
from urbanphotomeshqa.model import BuildingInvariantEncoder, MeshFaceEncoder
from train_decoupled_multimodal_quality import MultimodalQualityModel, evaluate, load_data


TEXTURE_ATTACKS = {"texture_detail_loss", "texture_region_missing", "texture_misalignment"}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def key(row):
    return str(row["asset_id"]), str(row["attack"]), str(row["level"])


def load_raw_geometry(audit_path: Path, data: dict, split: str) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    lookup = {key(row): Path(row["cache_path"]) for row in audit["records"] if row["split"] == split}
    sample_keys = []
    for asset_id, attack, level in zip(data["asset_ids"], data["attacks"], data["levels"]):
        sample_keys.append((asset_id, "clean", "clean") if attack in TEXTURE_ATTACKS
                           else (asset_id, attack, level))
    unique = {}
    for index, item in enumerate(sorted(set(sample_keys))):
        if item not in lookup:
            raise KeyError(f"Missing raw cache record: {split} {item}")
        with np.load(lookup[item]) as values:
            unique[item] = {"points": values["points"].astype(np.float32),
                            "graph": {name: values[name].copy()
                                      for name in ("face_features", "neighbors", "topology")}}
        if (index + 1) % 200 == 0:
            print(f"loaded raw {split} {index + 1}/{len(set(sample_keys))}", flush=True)
    return {"keys": sample_keys, "unique": unique}


def raw_batch(raw: dict, indices: np.ndarray, device: torch.device):
    items = [raw["unique"][raw["keys"][int(index)]] for index in indices]
    points = torch.from_numpy(np.stack([item["points"] for item in items])).to(device)
    graph = pad_mesh_graphs([item["graph"] for item in items])
    graph = {name: value.to(device) for name, value in graph.items()}
    return points, graph


class EndToEndModel(nn.Module):
    def __init__(self, point, mesh, quality, statistics):
        super().__init__(); self.point = point; self.mesh = mesh; self.quality = quality
        for branch in ("point_global", "mesh_global"):
            self.register_buffer(f"{branch}_mean", torch.tensor(statistics[branch]["mean"], dtype=torch.float32))
            self.register_buffer(f"{branch}_std", torch.tensor(statistics[branch]["std"], dtype=torch.float32))

    def normalize(self, name, value):
        return (value - getattr(self, f"{name}_mean")) / getattr(self, f"{name}_std")

    def forward(self, data, index, points, graph):
        batch = {
            "point_global": self.normalize("point_global", self.point(points)["global"]),
            "mesh_global": self.normalize("mesh_global", self.mesh(**graph)["global"]),
            "morphology": data["morphology"][index], "patches": data["patches"][index],
            "patch_mask": data["patch_mask"][index],
        }
        geometry = self.quality.geometry(batch, slice(None))
        texture = self.quality.texture.encode(
            data["tokens"][index], data["token_mask"][index],
            data["view_stats"][index], data["asset_stats"][index])
        output = self.quality.shared(torch.cat([geometry, texture], dim=1))
        return {"overall": output[:, 0], "geometry": output[:, 1], "texture": output[:, 2]}


def configure_stage(model, stage):
    for parameter in model.point.parameters(): parameter.requires_grad = False
    for parameter in model.mesh.parameters(): parameter.requires_grad = False
    if stage == "last_blocks":
        modules = [model.point.local_net[6], model.point.local_net[7],
                   model.mesh.face3, model.mesh.fuse]
        for module in modules:
            for parameter in module.parameters(): parameter.requires_grad = True
    return [parameter for encoder in (model.point, model.mesh)
            for parameter in encoder.parameters() if parameter.requires_grad]


@torch.no_grad()
def evaluate_end_to_end(model, data, raw, device, batch_size):
    model.eval(); predictions = {name: [] for name in ("overall", "geometry", "texture")}
    for start in range(0, len(data["overall"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data["overall"])))
        points, graph = raw_batch(raw, indices, device)
        output = model(data, torch.from_numpy(indices).to(device), points, graph)
        for name in predictions: predictions[name].append(output[name].cpu().numpy())
    predictions = {name: np.concatenate(value) for name, value in predictions.items()}
    targets = {"overall": data["overall"].cpu().numpy(),
               "geometry": data["geometry_target"].cpu().numpy(),
               "texture": data["texture_target"].cpu().numpy()}
    from train_real_gltf_quality import regression_metrics
    return {name: regression_metrics(predictions[name], targets[name]) for name in predictions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--mesh-checkpoint", type=Path, required=True)
    parser.add_argument("--quality-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("frozen_control", "last_blocks"), required=True)
    parser.add_argument("--epochs", type=int, default=60); parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--base-lr", type=float, default=1e-5); parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_all(args.seed); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    checkpoint = torch.load(args.quality_checkpoint, map_location=device, weights_only=False)
    statistics = {name: {key: np.asarray(value, np.float32) for key, value in item.items()}
                  for name, item in checkpoint["statistics"].items()}
    train, _ = load_data(args.feature_dir, "train", device, statistics)
    val, _ = load_data(args.feature_dir, "val", device, statistics)
    train_raw = load_raw_geometry(args.cache_audit, train, "train")
    val_raw = load_raw_geometry(args.cache_audit, val, "val")
    point_state = torch.load(args.point_checkpoint, map_location=device, weights_only=False)
    point = BuildingInvariantEncoder(**point_state["config"]["model"]).to(device); point.load_state_dict(point_state["model"])
    mesh_state = torch.load(args.mesh_checkpoint, map_location=device, weights_only=False)
    mesh = MeshFaceEncoder().to(device); mesh.load_state_dict(mesh_state["model"])
    quality = MultimodalQualityModel("shared", None).to(device); quality.load_state_dict(checkpoint["model"])
    model = EndToEndModel(point, mesh, quality, checkpoint["statistics"]).to(device)
    base_parameters = configure_stage(model, args.stage)
    groups = [{"params": model.quality.parameters(), "lr": args.head_lr}]
    if base_parameters: groups.append({"params": base_parameters, "lr": args.base_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    best_key = best_state = best_epoch = None; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.point.eval(); model.mesh.eval(); model.quality.train()
        if args.stage == "last_blocks":
            model.point.local_net[6:9].train(); model.mesh.face3.train(); model.mesh.fuse.train()
        order = torch.randperm(len(train["overall"]), generator=generator).numpy(); losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]; device_indices = torch.from_numpy(indices).to(device)
            points, graph = raw_batch(train_raw, indices, device); output = model(train, device_indices, points, graph)
            loss = (3.0 * F.smooth_l1_loss(output["overall"], train["overall"][device_indices])
                    + 1.5 * F.smooth_l1_loss(output["geometry"], train["geometry_target"][device_indices])
                    + 1.5 * F.smooth_l1_loss(output["texture"], train["texture_target"][device_indices]))
            delta = train["overall"][device_indices][:, None] - train["overall"][device_indices][None, :]
            valid = torch.abs(delta) > 0.03
            if valid.any():
                prediction_delta = output["overall"][:, None] - output["overall"][None, :]
                loss = loss + 0.2 * F.softplus(-5.0 * torch.sign(delta[valid]) * prediction_delta[valid]).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip); optimizer.step(); losses.append(float(loss.detach()))
        metrics = evaluate_end_to_end(model, val, val_raw, device, args.batch_size)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": metrics})
        key_value = (metrics["overall"]["srcc"], min(metrics["geometry"]["srcc"], metrics["texture"]["srcc"]),
                     -metrics["overall"]["mae"])
        if best_key is None or key_value > best_key:
            best_key, best_epoch, stale = key_value, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else: stale += 1
        print(f"{args.stage} epoch={epoch} loss={np.mean(losses):.4f} oqi={metrics['overall']['srcc']:.4f} "
              f"g={metrics['geometry']['srcc']:.4f} t={metrics['texture']['srcc']:.4f}", flush=True)
        if stale >= args.patience: break
    model.load_state_dict(best_state); metrics = evaluate_end_to_end(model, val, val_raw, device, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "status": "COMPLETE", "seed": args.seed, "stage": args.stage,
              "protocol": {"loaded_splits": ["train", "val"], "test_blind_loaded": False,
                           "selection": "Val OQI SRCC, min component SRCC, OQI MAE",
                           "head_lr": args.head_lr, "base_lr": args.base_lr,
                           "gradient_clip": args.gradient_clip, "patience": args.patience},
              "counts": {"train": len(train["overall"]), "val": len(val["overall"])},
              "sources": {"cache_audit": sha256_file(args.cache_audit),
                          "point_checkpoint": sha256_file(args.point_checkpoint),
                          "mesh_checkpoint": sha256_file(args.mesh_checkpoint),
                          "quality_checkpoint": sha256_file(args.quality_checkpoint)},
              "best_epoch": best_epoch, "epochs_run": len(history), "val": metrics, "history": history}
    torch.save({**result, "model": model.state_dict()}, args.output_dir / "best.pt")
    (args.output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": args.stage, "best_epoch": best_epoch, "val": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
