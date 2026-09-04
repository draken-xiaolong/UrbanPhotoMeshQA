#!/usr/bin/env python3
"""Audit cached versus fresh-glTF predictions for the frozen Iteration-2 ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from extract_real_gltf_cache import extract
from train_gatv2_mesh_quality import HierarchicalQualityGAT, QualityGAT
from urbanphotomeshqa.data import pad_mesh_graphs
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.local_features import extract_local_features
from urbanphotomeshqa.model import BuildingInvariantEncoder, MeshFaceEncoder
from urbanphotomeshqa.morphology import global_morphology_targets
from urbanphotomeshqa.texture_features import SpatialImageEncoder


def keyed(path: Path, names: tuple[str, ...]) -> dict:
    with np.load(path) as z:
        keys = zip(z["asset_ids"].astype(str), z["attacks"].astype(str), z["levels"].astype(str))
        return {key: {name: z[name][i].copy() for name in names} for i, key in enumerate(keys)}


def stats(base_path: Path, texture_path: Path, spatial_path: Path) -> dict:
    with np.load(base_path) as base, np.load(texture_path) as texture, np.load(spatial_path) as spatial:
        shared = {
            "point": [base["point_identity"].mean(0), np.maximum(base["point_identity"].std(0), 1e-5)],
            "morph": [base["morphology"].mean(0), np.maximum(base["morphology"].std(0), 1e-5)],
        }
        out = {"base": dict(shared), "spatial": dict(shared)}
        for source, prefix in ((texture, "base"), (spatial, "spatial")):
            out[prefix]["texture"] = [source["texture"].mean(0), np.maximum(source["texture"].std(0), 1e-5)]
            views = source["texture_views"].reshape(-1, source["texture_views"].shape[-1])
            out[prefix]["texture_views"] = [views.mean(0), np.maximum(views.std(0), 1e-5)]
        return out


def normalize(values: dict, norm: dict) -> dict:
    return {name: ((values[name] - norm[name][0]) / norm[name][1]).astype(np.float32)
            for name in ("point", "morph", "texture", "texture_views")}


def spatial_texture(local: dict) -> dict:
    patch = local["patch_mask"].astype(bool)
    view_mask = local["patch_view_mask"].astype(bool) & patch[:, None]
    count = np.maximum(view_mask.sum(1, keepdims=True), 1)
    view_token = (local["patch_view_tokens"] * view_mask[:, :, None]).sum(1) / count
    view_stats = (local["patch_view_stats"] * view_mask[:, :, None]).sum(1) / count
    atlas_mask = local["patch_atlas_mask"].astype(bool) & patch
    atlas_token = local["patch_atlas_tokens"] * atlas_mask[:, None]
    atlas_stats = local["patch_atlas_stats"] * atlas_mask[:, None]
    tokens = np.concatenate((view_token, view_stats, atlas_token, atlas_stats), 1).astype(np.float32)
    pooled = (tokens * patch[:, None]).sum(0) / max(int(patch.sum()), 1)
    return {"texture": pooled.astype(np.float32), "texture_views": tokens}


@torch.no_grad()
def predict(model, graph_np: dict, values: dict, device: torch.device) -> dict:
    graph = {k: v.to(device) for k, v in pad_mesh_graphs([graph_np]).items()}
    tensors = {k: torch.from_numpy(v[None]).to(device)
               for k, v in values.items()}
    return {k: v[0].detach().cpu().numpy() for k, v in model(
        graph, tensors["point"], tensors["morph"], tensors["texture"],
        tensors["texture_views"]).items()}


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-audit", type=Path, required=True)
    p.add_argument("--base-feature-dir", type=Path, required=True)
    p.add_argument("--texture-dir", type=Path, required=True)
    p.add_argument("--spatial-texture-dir", type=Path, required=True)
    p.add_argument("--point-checkpoint", type=Path, required=True)
    p.add_argument("--mesh-checkpoint", type=Path, required=True)
    p.add_argument("--models", type=Path, nargs=5, required=True)
    p.add_argument("--architectures", nargs=5, default=("gatv2", "hierarchical", "hierarchical", "hierarchical", "hierarchical"))
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--normalization", type=Path, help="Reuse normalization from a previous audit report")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(); device = torch.device(args.device)
    base = keyed(args.base_feature_dir / "features_val.npz", ("point_identity", "morphology"))
    texture = keyed(args.texture_dir / "strong_texture_val.npz", ("texture", "texture_views"))
    spatial = keyed(args.spatial_texture_dir / "strong_texture_val.npz", ("texture", "texture_views"))
    if args.normalization:
        saved = json.loads(args.normalization.read_text())["normalization"]
        norms = {group: {name: [np.asarray(pair["mean"], np.float32), np.asarray(pair["std"], np.float32)]
                         for name, pair in values.items()} for group, values in saved.items()}
    else:
        norms = stats(args.base_feature_dir / "features_train.npz",
                      args.texture_dir / "strong_texture_train.npz",
                      args.spatial_texture_dir / "strong_texture_train.npz")
    val_keys = set(base) & set(texture) & set(spatial)
    all_records = json.loads(args.cache_audit.read_text())["records"]
    records = [r for r in all_records if (r["asset_id"], r["attack"], r["level"]) in val_keys]
    rng = np.random.default_rng(args.seed)
    records = [records[i] for i in sorted(rng.choice(len(records), min(args.samples, len(records)), replace=False))]

    ps = torch.load(args.point_checkpoint, map_location=device, weights_only=False)
    point = BuildingInvariantEncoder(**ps["config"]["model"]).to(device).eval(); point.load_state_dict(ps["model"])
    ms = torch.load(args.mesh_checkpoint, map_location=device, weights_only=False)
    mesh = MeshFaceEncoder().to(device).eval(); mesh.load_state_dict(ms["model"])
    weights = EfficientNet_B0_Weights.DEFAULT
    image = efficientnet_b0(weights=weights).to(device).eval(); transform = weights.transforms()
    spatial_encoder = SpatialImageEncoder(device)
    models = []
    for architecture, path in zip(args.architectures, args.models):
        cls = QualityGAT if architecture == "gatv2" else HierarchicalQualityGAT
        texture_dim = 1280 if path != args.models[3] else spatial[next(iter(spatial))]["texture"].shape[0]
        model = cls(base[next(iter(base))]["point_identity"].shape[0], base[next(iter(base))]["morphology"].shape[0], texture_dim).to(device).eval()
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"]); models.append(model)

    lookup = {(r["asset_id"], r["attack"], r["level"]): r for r in records}
    available = sorted(set(lookup) & set(base) & set(texture) & set(spatial))
    rows = []
    for key in available:
        record = lookup[key]
        fresh, _ = extract(Path(record["gltf_path"]), record["extractor_settings"])
        points = torch.from_numpy(fresh["points"][None]).to(device)
        graph = {name: fresh[name] for name in ("face_features", "neighbors", "topology")}
        graph_t = {k: v.to(device) for k, v in pad_mesh_graphs([graph]).items()}
        point_value = F.normalize(point(points)["identity"], dim=1)[0].cpu().numpy().astype(np.float32)
        _ = mesh(**graph_t)  # verifies the frozen mesh encoder accepts the fresh graph.
        view_tensor = torch.stack([transform(Image.fromarray(v)) for v in fresh["render_views"]]).to(device)
        view = F.normalize(image.avgpool(image.features(view_tensor)).flatten(1), dim=1)
        fresh_base = {"point": point_value,
                      "morph": global_morphology_targets(points)[0].cpu().numpy().astype(np.float32),
                      "texture": F.normalize(view.mean(0, keepdim=True), dim=1)[0].cpu().numpy().astype(np.float32),
                      "texture_views": view.cpu().numpy().astype(np.float32)}
        cached_base = {"point": base[key]["point_identity"], "morph": base[key]["morphology"], **texture[key]}
        local = extract_local_features(GltfReader(Path(record["gltf_path"])).load_mesh(include_texture=True), spatial_encoder, render_size=224)
        fresh_spatial = spatial_texture(local); cached_spatial = spatial[key]
        predictions = []; deploy_predictions = []; component_predictions = []
        for source_base, source_spatial in ((cached_base, cached_spatial), (fresh_base, fresh_spatial)):
            normal_base = normalize(source_base, norms["base"]); normal_spatial = normalize({**source_base, **source_spatial}, norms["spatial"])
            outputs = [predict(model, graph, normal_spatial if i == 3 else normal_base, device)
                       for i, model in enumerate(models)]
            component_predictions.append(outputs)
            predictions.append({name: sum(out[name] for out in outputs) / len(outputs)
                                for name in outputs[0]})
            deploy = [outputs[i] for i in (0, 1, 2, 4)]
            deploy_predictions.append({name: sum(out[name] for out in deploy) / len(deploy)
                                       for name in deploy[0]})
        differences = {name: float(np.max(np.abs(predictions[0][name] - predictions[1][name])))
                       for name in ("overall", "geometry", "texture", "severity", "attack", "ordinal")}
        deploy_differences = {name: float(np.max(np.abs(deploy_predictions[0][name] - deploy_predictions[1][name])))
                              for name in ("overall", "geometry", "texture", "severity", "attack", "ordinal")}
        feature_differences = {
            **{name: float(np.max(np.abs(cached_base[name] - fresh_base[name])))
               for name in cached_base},
            **{f"spatial_{name}": float(np.max(np.abs(cached_spatial[name] - fresh_spatial[name])))
               for name in cached_spatial},
        }
        component_differences = [{name: float(np.max(np.abs(component_predictions[0][i][name] -
                                                               component_predictions[1][i][name])))
                                  for name in ("overall", "geometry", "texture", "severity", "attack", "ordinal")}
                                 for i in range(len(models))]
        rows.append({"key": key, "differences": differences, "deployable_differences": deploy_differences,
                     "component_differences": component_differences,
                     "feature_differences": feature_differences})
        print(f"audit {len(rows)}/{len(available)} max={max(differences.values()):.7f}", flush=True)
    maxima = {name: max(row["deployable_differences"][name] for row in rows) for name in rows[0]["deployable_differences"]}
    thresholds = {"quality_outputs": 5e-4, "auxiliary_outputs": 5e-3}
    passed = max(maxima[name] for name in ("overall", "geometry", "texture", "severity")) <= thresholds["quality_outputs"]
    passed = passed and max(maxima[name] for name in ("attack", "ordinal")) <= thresholds["auxiliary_outputs"]
    report = {"schema_version": 1, "passed": passed, "samples": len(rows), "thresholds": thresholds,
              "max_prediction_absolute_difference": maxima,
              "protocol": "fresh glTF versus training caches; deployable models 1/2/3/5; Val-only deterministic sample; Test/Blind not loaded",
              "excluded_model": "spatial Patch texture model excluded because its online features are not reproducible",
              "normalization": {group: {name: {"mean": pair[0].tolist(), "std": pair[1].tolist()}
                                         for name, pair in values.items()} for group, values in norms.items()},
              "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": passed, "samples": len(rows), "maxima": maxima}))
    if not passed: raise SystemExit(1)


if __name__ == "__main__":
    main()
