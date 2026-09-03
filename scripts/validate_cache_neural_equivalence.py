#!/usr/bin/env python3
"""Verify cached and fresh-glTF inputs produce numerically identical Base features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from urbanphotomeshqa.data import pad_mesh_graphs
from urbanphotomeshqa.model import BuildingInvariantEncoder, MeshFaceEncoder
from urbanphotomeshqa.morphology import global_morphology_targets
from urbanphotomeshqa.render_features import ImageEncoder

from extract_real_gltf_cache import extract
from infer_real_gltf_quality import load_quality_head, normalize


def load_encoders(point_checkpoint: Path, mesh_checkpoint: Path, device):
    point_state = torch.load(point_checkpoint, map_location=device, weights_only=False)
    point = BuildingInvariantEncoder(**point_state["config"]["model"]).to(device).eval()
    point.load_state_dict(point_state["model"])
    mesh_state = torch.load(mesh_checkpoint, map_location=device, weights_only=False)
    mesh = MeshFaceEncoder().to(device).eval()
    mesh.load_state_dict(mesh_state["model"])
    return point, mesh


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.flatten().float(), b.flatten().float()
    return {"cosine": float(F.cosine_similarity(a[None], b[None]).item()),
            "max_abs": float(torch.max(torch.abs(a - b)).item())}


@torch.no_grad()
def encode(arrays, point_encoder, mesh_encoder, image_encoder, device):
    points = torch.from_numpy(arrays["points"][None]).to(device)
    graph = pad_mesh_graphs([{"face_features": arrays["face_features"],
                             "neighbors": arrays["neighbors"], "topology": arrays["topology"]}])
    graph = {key: value.to(device) for key, value in graph.items()}
    return {
        "point": F.normalize(point_encoder(points)["identity"], dim=1).cpu(),
        "mesh": F.normalize(mesh_encoder(**graph)["identity"], dim=1).cpu(),
        "morphology": global_morphology_targets(points).cpu(),
        "texture": torch.from_numpy(image_encoder(list(arrays["render_views"]))["pooled"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--mesh-checkpoint", type=Path, required=True)
    parser.add_argument("--quality-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal neural cache validation")
    source = json.loads(args.cache_audit.read_text(encoding="utf-8"))
    point, mesh = load_encoders(args.point_checkpoint, args.mesh_checkpoint, device)
    image = ImageEncoder(device)
    quality_state, quality_model = (load_quality_head(args.quality_checkpoint, device)
                                    if args.quality_checkpoint else (None, None))
    results, errors = [], []
    for index, record in enumerate(source["records"]):
        with np.load(record["cache_path"]) as values:
            cached = {name: values[name].copy() for name in
                      ("points", "face_features", "neighbors", "topology", "render_views",
                       "patches", "patch_mask")}
        fresh, _ = extract(Path(record["gltf_path"]), record["extractor_settings"])
        cached_features = encode(cached, point, mesh, image, device)
        fresh_features = encode(fresh, point, mesh, image, device)
        checks = {branch: compare(cached_features[branch], fresh_features[branch])
                  for branch in cached_features}
        prediction_checks = None
        if quality_model is not None:
            predictions = []
            for arrays, features in ((cached, cached_features), (fresh, fresh_features)):
                quality_values = {name: features[name].numpy().reshape(-1) for name in features}
                quality_values.update({"patches": arrays["patches"], "patch_mask": arrays["patch_mask"]})
                inputs = normalize(quality_values, quality_state, device)
                predictions.append(quality_model(*inputs))
            prediction_checks = {
                name: float(torch.max(torch.abs(predictions[0][name] - predictions[1][name])).item())
                for name in ("attack", "severity", "overall", "geometry", "texture")
            }
        ok = all(value["cosine"] >= 0.999999 and value["max_abs"] <= 1e-5 for value in checks.values())
        ok = ok and (prediction_checks is None or max(prediction_checks.values()) <= 1e-4)
        if not ok:
            errors.append(record["cache_path"])
        results.append({"asset_id": record["asset_id"], "attack": record["attack"],
                        "level": record["level"], "ok": ok, "branches": checks,
                        "final_prediction_max_abs": prediction_checks})
        print(f"ok={ok} {index + 1}/{len(source['records'])} {record['asset_id']} {record['attack']} {record['level']}", flush=True)
    report = {"schema_version": 1, "passed": not errors, "thresholds": {
        "cosine_min": 0.999999, "max_abs_max": 1e-5,
        "final_prediction_difference_max": 1e-4,
        "final_prediction_check": "performed" if quality_model is not None else "not requested",
    }, "errors": errors, "records": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "records": len(results), "errors": errors}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
