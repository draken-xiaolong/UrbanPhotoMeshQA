#!/usr/bin/env python3
"""Partially fine-tune Point/Mesh Base modules for geometry quality on Train/Val only."""

from __future__ import annotations

import argparse
import hashlib
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
from urbanphotomeshqa.morphology import global_morphology_targets
from train_real_gltf_quality import QualityHead, regression_metrics


GEOMETRY_ATTACKS = {"clean", "geometry_hole", "mesh_simplification_qem",
                    "geometry_noise_spike"}


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_key(row) -> tuple[str, str, str]:
    return str(row["asset_id"]), str(row["attack"]), str(row["level"])


def load_target_lookup(root: Path, split: str) -> dict[tuple[str, str, str], tuple[float, float]]:
    path = root / f"objective_targets_v2_{split}.npz"
    if not path.is_file():
        path = root / f"objective_targets_{split}.npz"
    with np.load(path) as values:
        keys = zip(values["asset_ids"].astype(str), values["attacks"].astype(str),
                   values["levels"].astype(str))
        return {key: (float(overall), float(geometry)) for key, overall, geometry in zip(
            keys, values["overall_quality"], values["geometry_quality"])}


def load_split(records: list[dict], target_root: Path, split: str) -> dict:
    selected = [row for row in records if row["split"] == split and row["attack"] in GEOMETRY_ATTACKS]
    targets = load_target_lookup(target_root, split)
    points, graphs, overall, geometry, keys = [], [], [], [], []
    for index, row in enumerate(selected):
        path = Path(row["cache_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as values:
            points.append(values["points"].astype(np.float32))
            graphs.append({name: values[name].copy() for name in ("face_features", "neighbors", "topology")})
        key = sample_key(row)
        if key not in targets:
            raise KeyError(f"Missing objective target: {split} {key}")
        oqi, geometry_quality = targets[key]
        overall.append(oqi); geometry.append(geometry_quality); keys.append(key)
        if (index + 1) % 200 == 0:
            print(f"loaded {split} {index + 1}/{len(selected)}", flush=True)
    return {"points": np.stack(points), "graphs": graphs,
            "overall": np.asarray(overall, np.float32),
            "geometry": np.asarray(geometry, np.float32), "keys": keys}


class FineTunedGeometryModel(nn.Module):
    def __init__(self, point: nn.Module, mesh: nn.Module, head: QualityHead,
                 statistics: dict[str, dict[str, list[float]]]):
        super().__init__(); self.point = point; self.mesh = mesh; self.head = head
        for branch in ("point", "mesh", "morphology", "texture"):
            self.register_buffer(f"{branch}_center", torch.tensor(statistics[branch]["mean"], dtype=torch.float32))
            self.register_buffer(f"{branch}_scale", torch.tensor(statistics[branch]["std"], dtype=torch.float32))

    def normalize(self, branch: str, value: torch.Tensor) -> torch.Tensor:
        return (value - getattr(self, f"{branch}_center")) / getattr(self, f"{branch}_scale")

    def forward(self, points: torch.Tensor, graph: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        point = self.normalize("point", self.point(points)["global"])
        mesh = self.normalize("mesh", self.mesh(**graph)["global"])
        morphology = self.normalize("morphology", global_morphology_targets(points))
        texture = self.normalize("texture", torch.zeros((len(points), 1), device=points.device))
        patches = torch.zeros((len(points), 1, 58), device=points.device)
        patch_mask = torch.ones((len(points), 1), dtype=torch.bool, device=points.device)
        return self.head([point, mesh, morphology, texture], patches, patch_mask)


def configure_stage(model: FineTunedGeometryModel, stage: str) -> list[nn.Parameter]:
    for parameter in model.point.parameters(): parameter.requires_grad = False
    for parameter in model.mesh.parameters(): parameter.requires_grad = False
    modules = [] if stage == "frozen_control" else [model.mesh.fuse]
    if stage == "last_blocks":
        modules.extend([model.point.local_net[6], model.point.local_net[7], model.mesh.face3])
    for module in modules:
        for parameter in module.parameters(): parameter.requires_grad = True
    return [parameter for encoder in (model.point, model.mesh)
            for parameter in encoder.parameters() if parameter.requires_grad]


def batch_tensors(data: dict, indices: np.ndarray, device: torch.device):
    points = torch.from_numpy(data["points"][indices]).to(device)
    graph = pad_mesh_graphs([data["graphs"][int(index)] for index in indices])
    graph = {name: value.to(device) for name, value in graph.items()}
    overall = torch.from_numpy(data["overall"][indices]).to(device)
    geometry = torch.from_numpy(data["geometry"][indices]).to(device)
    return points, graph, overall, geometry


@torch.no_grad()
def evaluate(model, data: dict, device: torch.device, batch_size: int) -> dict:
    model.eval(); estimates = []
    for start in range(0, len(data["overall"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data["overall"])))
        points, graph, _, _ = batch_tensors(data, indices, device)
        output = model(points, graph)
        estimates.append(np.stack([output["overall"].cpu().numpy(),
                                   output["geometry"].cpu().numpy()], axis=1))
    estimates = np.concatenate(estimates)
    return {"overall": regression_metrics(estimates[:, 0], data["overall"]),
            "geometry": regression_metrics(estimates[:, 1], data["geometry"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--objective-target-dir", type=Path, required=True)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--mesh-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("frozen_control", "mesh_fuse", "last_blocks"), required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    audit = json.loads(args.cache_audit.read_text(encoding="utf-8"))
    train = load_split(audit["records"], args.objective_target_dir, "train")
    val = load_split(audit["records"], args.objective_target_dir, "val")
    point_state = torch.load(args.point_checkpoint, map_location=device, weights_only=False)
    point = BuildingInvariantEncoder(**point_state["config"]["model"]).to(device)
    point.load_state_dict(point_state["model"])
    mesh_state = torch.load(args.mesh_checkpoint, map_location=device, weights_only=False)
    mesh = MeshFaceEncoder().to(device); mesh.load_state_dict(mesh_state["model"])
    head_state = torch.load(args.head_checkpoint, map_location=device, weights_only=False)
    if head_state.get("base_representation") != "global":
        raise ValueError("--head-checkpoint must be the frozen global-feature candidate")
    head = QualityHead(head_state["dims"], head_state["branch_indices"],
                       head_state["use_patches"]).to(device)
    head.load_state_dict(head_state["model"])
    model = FineTunedGeometryModel(point, mesh, head, head_state["statistics"]["branches"]).to(device)
    base_parameters = configure_stage(model, args.stage)
    parameter_groups = [{"params": model.head.parameters(), "lr": args.head_lr}]
    if base_parameters:
        parameter_groups.append({"params": base_parameters, "lr": args.base_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)

    history, best_key, best_state, best_epoch, stale = [], None, None, None, 0
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.point.eval(); model.mesh.eval(); model.head.train()
        if args.stage == "last_blocks": model.point.local_net[6:9].train()
        model.mesh.fuse.train()
        if args.stage == "last_blocks": model.mesh.face3.train()
        order = torch.randperm(len(train["overall"]), generator=generator).numpy()
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            points, graph, overall, geometry = batch_tensors(train, indices, device)
            output = model(points, graph)
            loss = (3.0 * F.smooth_l1_loss(output["overall"], overall)
                    + 1.5 * F.smooth_l1_loss(output["geometry"], geometry))
            delta = overall[:, None] - overall[None, :]
            valid = torch.abs(delta) > 0.05
            if valid.any():
                prediction_delta = output["overall"][:, None] - output["overall"][None, :]
                loss = loss + 0.2 * F.softplus(
                    -5.0 * torch.sign(delta[valid]) * prediction_delta[valid]).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); losses.append(float(loss.detach()))
        metrics = evaluate(model, val, device, args.batch_size)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": metrics})
        key = (metrics["overall"]["srcc"], metrics["geometry"]["srcc"],
               -metrics["overall"]["mae"])
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        print(f"{args.stage} epoch={epoch} loss={np.mean(losses):.4f} "
              f"val_mae={metrics['overall']['mae']:.4f} val_srcc={metrics['overall']['srcc']:.4f}",
              flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, val, device, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"schema_version": 1, "seed": args.seed, "stage": args.stage,
                  "protocol": {"loaded_splits": ["train", "val"], "test_blind_loaded": False,
                               "selection": "Val OQI SRCC, Geometry SRCC, OQI MAE",
                               "gradient_clip": args.gradient_clip,
                               "head_lr": args.head_lr, "base_lr": args.base_lr,
                               "patience": args.patience},
                  "counts": {"train": len(train["overall"]), "val": len(val["overall"])},
                  "sources": {"cache_audit": sha256_file(args.cache_audit),
                              "point_checkpoint": sha256_file(args.point_checkpoint),
                              "mesh_checkpoint": sha256_file(args.mesh_checkpoint),
                              "head_checkpoint": sha256_file(args.head_checkpoint)},
                  "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                              if parameter.requires_grad),
                  "best_epoch": best_epoch, "epochs_run": len(history),
                  "val": final_metrics, "history": history}
    torch.save({**provenance, "model": model.state_dict()}, args.output_dir / "best.pt")
    (args.output_dir / "results.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: provenance[key] for key in ("stage", "best_epoch", "epochs_run", "val")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
