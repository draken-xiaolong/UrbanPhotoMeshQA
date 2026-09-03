#!/usr/bin/env python3
"""No-reference quality inference for one real file-native glTF package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from extract_real_gltf_cache import extract
from train_real_gltf_quality import ATTACKS, BRANCHES, QualityHead
from urbanphotomeshqa.data import pad_mesh_graphs
from urbanphotomeshqa.model import BuildingInvariantEncoder, MeshFaceEncoder
from urbanphotomeshqa.morphology import global_morphology_targets
from urbanphotomeshqa.render_features import ImageEncoder


def load_quality_head(path: Path, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=False)
    model = QualityHead(state["dims"], state["branch_indices"], state["use_patches"]).to(device).eval()
    model.load_state_dict(state["model"])
    return state, model


def normalize(values, state, device):
    statistics = state["statistics"]
    branches = []
    for name in BRANCHES:
        mean = np.asarray(statistics["branches"][name]["mean"], np.float32)
        std = np.asarray(statistics["branches"][name]["std"], np.float32)
        branches.append(torch.from_numpy(((values[name] - mean) / std)[None]).to(device))
    patch_mean = np.asarray(statistics["patch_mean"], np.float32)
    patch_std = np.asarray(statistics["patch_std"], np.float32)
    patches = (values["patches"] - patch_mean) / patch_std
    patches[~values["patch_mask"]] = 0.0
    return branches, torch.from_numpy(patches[None].astype(np.float32)).to(device), \
        torch.from_numpy(values["patch_mask"][None]).to(device)


def calibrated(name, value, calibration):
    parameters = calibration.get("calibration", {}).get(name)
    if parameters is None:
        return float(np.clip(value, 0.0, 1.0))
    return float(np.clip(parameters["slope"] * value + parameters["intercept"], 0.0, 1.0))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gltf", type=Path)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--mesh-checkpoint", type=Path, required=True)
    parser.add_argument("--quality-checkpoint", type=Path, required=True)
    parser.add_argument("--patch-checkpoint", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required for formal inference")

    settings = {"schema_version": 1, "points": 1024, "patches": 16,
                "patch_neighbors": 32, "render_size": 96,
                "render_directions": 6, "seed": 2026}
    raw, mesh_details = extract(args.gltf, settings)

    point_state = torch.load(args.point_checkpoint, map_location=device, weights_only=False)
    point_encoder = BuildingInvariantEncoder(**point_state["config"]["model"]).to(device).eval()
    point_encoder.load_state_dict(point_state["model"])
    mesh_state = torch.load(args.mesh_checkpoint, map_location=device, weights_only=False)
    mesh_encoder = MeshFaceEncoder().to(device).eval()
    mesh_encoder.load_state_dict(mesh_state["model"])
    points = torch.from_numpy(raw["points"][None]).to(device)
    graph_np = {name: raw[name] for name in ("face_features", "neighbors", "topology")}
    graph = {name: value.to(device) for name, value in pad_mesh_graphs([graph_np]).items()}
    values = {
        "point": F.normalize(point_encoder(points)["identity"], dim=1)[0].cpu().numpy(),
        "mesh": F.normalize(mesh_encoder(**graph)["identity"], dim=1)[0].cpu().numpy(),
        "morphology": global_morphology_targets(points)[0].cpu().numpy(),
        "texture": ImageEncoder(device)(list(raw["render_views"]))["pooled"],
        "patches": raw["patches"], "patch_mask": raw["patch_mask"],
    }

    state, model = load_quality_head(args.quality_checkpoint, device)
    branches, patches, patch_mask = normalize(values, state, device)
    prediction = model(branches, patches, patch_mask)
    probabilities = torch.softmax(prediction["attack"], 1)[0].cpu().numpy()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else {}
    scores = {name: calibrated(name, float(prediction[name][0]), calibration)
              for name in ("overall", "geometry", "texture")}
    report = {
        "schema_version": 1,
        "protocol": "single-model no-reference inference; no pristine model, alignment, or difference input",
        "gltf": str(args.gltf.resolve()),
        "mesh": mesh_details,
        "quality": {
            "overall_oqi": scores["overall"],
            "overall_score_0_100": 100.0 * scores["overall"],
            "geometry_quality": scores["geometry"],
            "texture_quality": scores["texture"],
            "predicted_degradation_strength": float(prediction["severity"][0]),
        },
    }
    if not state.get("quality_only", False):
        report["auxiliary_degradation_probabilities"] = {
            name: float(value) for name, value in zip(ATTACKS, probabilities)
        }
    if args.patch_checkpoint:
        patch_state, patch_model = load_quality_head(args.patch_checkpoint, device)
        patch_branches, patch_values, patch_valid = normalize(values, patch_state, device)
        patch_prediction = patch_model(patch_branches, patch_values, patch_valid)
        valid = values["patch_mask"]
        report["local_patch_quality"] = {
            "scores": patch_prediction["patch_quality"][0].cpu().numpy()[valid].astype(float).tolist(),
            "attention": patch_prediction["patch_weights"][0].cpu().numpy()[valid].astype(float).tolist(),
            "interpretation": "lower quality means a more suspicious local geometric patch",
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
