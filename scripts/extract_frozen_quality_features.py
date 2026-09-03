#!/usr/bin/env python3
"""Encode cached real-glTF inputs with the frozen four-branch Base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from urbanphotomeshqa.data import pad_mesh_graphs
from urbanphotomeshqa.integrity import extractor_signature, sha256_file
from urbanphotomeshqa.model import BuildingInvariantEncoder, MeshFaceEncoder
from urbanphotomeshqa.morphology import global_morphology_targets
from urbanphotomeshqa.render_features import ImageEncoder


BRANCHES = ("point", "mesh", "morphology", "texture")


def key(row):
    return row["asset_id"], row["attack"], row["level"]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--mesh-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required for formal frozen feature extraction")
    point_state = torch.load(args.point_checkpoint, map_location=device, weights_only=False)
    point_encoder = BuildingInvariantEncoder(**point_state["config"]["model"]).to(device).eval()
    point_encoder.load_state_dict(point_state["model"])
    mesh_state = torch.load(args.mesh_checkpoint, map_location=device, weights_only=False)
    mesh_encoder = MeshFaceEncoder().to(device).eval(); mesh_encoder.load_state_dict(mesh_state["model"])
    image_encoder = ImageEncoder(device)
    signature_payload = {"schema_version": 1, "point_checkpoint": sha256_file(args.point_checkpoint),
                         "mesh_checkpoint": sha256_file(args.mesh_checkpoint),
                         "texture_encoder": "torchvision_mobilenet_v3_small_default",
                         "branches": BRANCHES}
    signature = extractor_signature(signature_payload)
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    targets = {key(row): row for row in dataset["records"]}
    cache_records = json.loads(args.cache_audit.read_text(encoding="utf-8"))["records"]
    output = {split: [] for split in ("train", "val", "test", "blind")}
    for index, record in enumerate(cache_records):
        target = targets[key(record)]
        raw_path = Path(record["cache_path"])
        neural_path = raw_path.with_name("neural_features.npz")
        values = None
        if neural_path.is_file():
            try:
                with np.load(neural_path) as stored:
                    if str(stored["signature"].item()) == signature:
                        values = {name: stored[name].copy() for name in (*BRANCHES, "patches", "patch_mask")}
            except Exception:
                values = None
        if values is None:
            with np.load(raw_path) as raw:
                points_np = raw["points"].astype(np.float32)
                graph_np = {name: raw[name].copy() for name in ("face_features", "neighbors", "topology")}
                views = raw["render_views"].copy()
                patches, patch_mask = raw["patches"].copy(), raw["patch_mask"].copy()
            points = torch.from_numpy(points_np[None]).to(device)
            graph = pad_mesh_graphs([graph_np]); graph = {name: value.to(device) for name, value in graph.items()}
            values = {
                "point": F.normalize(point_encoder(points)["identity"], dim=1)[0].cpu().numpy().astype(np.float32),
                "mesh": F.normalize(mesh_encoder(**graph)["identity"], dim=1)[0].cpu().numpy().astype(np.float32),
                "morphology": global_morphology_targets(points)[0].cpu().numpy().astype(np.float32),
                "texture": image_encoder(list(views))["pooled"].astype(np.float32),
                "patches": patches.astype(np.float32), "patch_mask": patch_mask,
            }
            np.savez_compressed(neural_path, **values, signature=np.asarray(signature))
        output[target["split"]].append({"target": target, "values": values})
        print(f"ok {index + 1}/{len(cache_records)} {target['split']} {target['asset_id']} {target['attack']} {target['level']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        np.savez_compressed(
            args.output_dir / f"features_{split}.npz",
            asset_ids=np.asarray([row["target"]["asset_id"] for row in rows]),
            attacks=np.asarray([row["target"]["attack"] for row in rows]),
            levels=np.asarray([row["target"]["level"] for row in rows]),
            attack_index=np.asarray([row["target"]["attack_index"] for row in rows], np.int64),
            severity=np.asarray([row["target"]["severity"] for row in rows], np.float32),
            overall_quality=np.asarray([row["target"]["overall_quality"] for row in rows], np.float32),
            geometry_quality=np.asarray([row["target"]["geometry_quality"] for row in rows], np.float32),
            texture_quality=np.asarray([row["target"]["texture_quality"] for row in rows], np.float32),
            **{branch: np.stack([row["values"][branch] for row in rows]) for branch in BRANCHES},
            patches=np.stack([row["values"]["patches"] for row in rows]),
            patch_mask=np.stack([row["values"]["patch_mask"] for row in rows]),
        )
    metadata = {"schema_version": 1, "signature": signature, "signature_payload": signature_payload,
                "counts": {split: len(rows) for split, rows in output.items()},
                "dimensions": {branch: int(output["train"][0]["values"][branch].shape[-1]) for branch in BRANCHES},
                "patch_shape": list(output["train"][0]["values"]["patches"].shape)}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", **metadata}))


if __name__ == "__main__":
    main()
