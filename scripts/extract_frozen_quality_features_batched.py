#!/usr/bin/env python3
"""Batch GPU encoder for Iteration 2 raw glTF caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from urbanphotomeshqa.data import pad_mesh_graphs
from urbanphotomeshqa.model import BuildingInvariantEncoder, MeshFaceEncoder
from urbanphotomeshqa.morphology import global_morphology_targets
from urbanphotomeshqa.render_features import ImageEncoder


BRANCHES = ("point_identity", "point_global", "mesh_identity", "mesh_global",
            "morphology", "texture")


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
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    point_state = torch.load(args.point_checkpoint, map_location=device, weights_only=False)
    point = BuildingInvariantEncoder(**point_state["config"]["model"]).to(device).eval()
    point.load_state_dict(point_state["model"])
    mesh_state = torch.load(args.mesh_checkpoint, map_location=device, weights_only=False)
    mesh = MeshFaceEncoder().to(device).eval()
    mesh.load_state_dict(mesh_state["model"])
    image = ImageEncoder(device)

    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    targets = {key(row): row for row in dataset["records"]}
    audit = json.loads(args.cache_audit.read_text(encoding="utf-8"))["records"]
    selected = [row for row in audit if key(row) in targets
                and targets[key(row)]["split"] in set(args.splits)]
    output = {split: [] for split in args.splits}
    for start in range(0, len(selected), args.batch_size):
        batch_records = selected[start:start + args.batch_size]
        points_np, graphs, views_np, patches_np, masks_np = [], [], [], [], []
        for record in batch_records:
            with np.load(record["cache_path"]) as raw:
                points_np.append(raw["points"].astype(np.float32))
                graphs.append({name: raw[name].copy()
                               for name in ("face_features", "neighbors", "topology")})
                views_np.append(raw["render_views"].copy())
                patches_np.append(raw["patches"].copy())
                masks_np.append(raw["patch_mask"].copy())
        points_tensor = torch.from_numpy(np.stack(points_np)).to(device)
        graph = {name: value.to(device) for name, value in pad_mesh_graphs(graphs).items()}
        point_out = point(points_tensor)
        mesh_out = mesh(**graph)
        view_tensors = torch.stack([
            image.transform(Image.fromarray(view))
            for views in views_np for view in views
        ]).to(device)
        embeddings = F.normalize(image.pool(image.features(view_tensors)).flatten(1), dim=1)
        texture = F.normalize(embeddings.reshape(len(batch_records), -1, embeddings.shape[-1]).mean(1), dim=1)
        morphology = global_morphology_targets(points_tensor)
        for index, record in enumerate(batch_records):
            target = targets[key(record)]
            values = {
                "point_identity": point_out["identity"][index].cpu().numpy().astype(np.float32),
                "point_global": point_out["global"][index].cpu().numpy().astype(np.float32),
                "mesh_identity": mesh_out["identity"][index].cpu().numpy().astype(np.float32),
                "mesh_global": mesh_out["global"][index].cpu().numpy().astype(np.float32),
                "morphology": morphology[index].cpu().numpy().astype(np.float32),
                "texture": texture[index].cpu().numpy().astype(np.float32),
                "patches": patches_np[index].astype(np.float32),
                "patch_mask": masks_np[index],
            }
            output[target["split"]].append({"target": target, "values": values})
        print(f"batch {min(start + len(batch_records), len(selected))}/{len(selected)}", flush=True)

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
    metadata = {"schema_version": 1, "method": "batched equivalent frozen four-branch encoder",
                "batch_size": args.batch_size, "counts": {k: len(v) for k, v in output.items()}}
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
