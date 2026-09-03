#!/usr/bin/env python3
"""Encode cached real-glTF inputs with the frozen four-branch Base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from urbanphotomeshqa.data import pad_mesh_graphs
from urbanphotomeshqa.integrity import extractor_signature, sha256_file
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
    parser.add_argument("--feature-cache-dir", type=Path,
                        help="Central resumable cache; defaults inside output-dir")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test", "blind"),
                        default=("train", "val", "test", "blind"))
    parser.add_argument("--attacks", nargs="+",
                        choices=("clean", "geometry_hole", "mesh_simplification_qem",
                                 "geometry_noise_spike", "texture_detail_loss",
                                 "texture_region_missing", "texture_misalignment"))
    parser.add_argument("--skip-texture", action="store_true",
                        help="Emit a zero placeholder for geometry-only Base ablations")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required for formal frozen feature extraction")
    point_state = torch.load(args.point_checkpoint, map_location=device, weights_only=False)
    point_encoder = BuildingInvariantEncoder(**point_state["config"]["model"]).to(device).eval()
    point_encoder.load_state_dict(point_state["model"])
    mesh_state = torch.load(args.mesh_checkpoint, map_location=device, weights_only=False)
    mesh_encoder = MeshFaceEncoder().to(device).eval(); mesh_encoder.load_state_dict(mesh_state["model"])
    image_encoder = None if args.skip_texture else ImageEncoder(device)
    repository = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), repository / "src/urbanphotomeshqa/model.py",
                    repository / "src/urbanphotomeshqa/data.py",
                    repository / "src/urbanphotomeshqa/morphology.py",
                    repository / "src/urbanphotomeshqa/render_features.py"]
    signature_payload = {"schema_version": 2, "point_checkpoint": sha256_file(args.point_checkpoint),
                         "mesh_checkpoint": sha256_file(args.mesh_checkpoint),
                         "texture_encoder": ("skipped_geometry_only" if args.skip_texture else
                                             "torchvision_mobilenet_v3_small_default"),
                         "branches": BRANCHES,
                         "source_sha256": {str(path.relative_to(repository)): sha256_file(path)
                                           for path in source_paths}}
    signature = extractor_signature(signature_payload)
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    targets = {key(row): row for row in dataset["records"]}
    cache_records = json.loads(args.cache_audit.read_text(encoding="utf-8"))["records"]
    selected_splits = tuple(args.splits)
    selected_attacks = set(args.attacks or [row["attack"] for row in dataset["records"]])
    selected = [record for record in cache_records
                if key(record) in targets
                and targets[key(record)]["split"] in selected_splits
                and targets[key(record)]["attack"] in selected_attacks]
    selected = selected[args.shard_index::args.shard_count]
    output = {split: [] for split in selected_splits}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache_dir or (args.output_dir / "per_asset_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(selected):
        target = targets[key(record)]
        raw_path = Path(record["cache_path"])
        cache_key = extractor_signature({"sample": key(record)})
        neural_path = cache_dir / f"{cache_key}.npz"
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
                views = None if args.skip_texture else raw["render_views"].copy()
                patches, patch_mask = raw["patches"].copy(), raw["patch_mask"].copy()
            points = torch.from_numpy(points_np[None]).to(device)
            graph = pad_mesh_graphs([graph_np]); graph = {name: value.to(device) for name, value in graph.items()}
            point_output = point_encoder(points)
            mesh_output = mesh_encoder(**graph)
            values = {
                "point_identity": point_output["identity"][0].cpu().numpy().astype(np.float32),
                "point_global": point_output["global"][0].cpu().numpy().astype(np.float32),
                "mesh_identity": mesh_output["identity"][0].cpu().numpy().astype(np.float32),
                "mesh_global": mesh_output["global"][0].cpu().numpy().astype(np.float32),
                "morphology": global_morphology_targets(points)[0].cpu().numpy().astype(np.float32),
                "texture": (np.zeros(1, dtype=np.float32) if args.skip_texture else
                            image_encoder(list(views))["pooled"].astype(np.float32)),
                "patches": patches.astype(np.float32), "patch_mask": patch_mask,
            }
            np.savez_compressed(neural_path, **values, signature=np.asarray(signature))
        output[target["split"]].append({"target": target, "values": values})
        print(f"ok {index + 1}/{len(selected)} {target['split']} {target['asset_id']} {target['attack']} {target['level']}", flush=True)
    for split, rows in output.items():
        if not rows:
            raise ValueError(f"No selected records for split: {split}")
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
    metadata = {"schema_version": 2, "signature": signature, "signature_payload": signature_payload,
                "counts": {split: len(rows) for split, rows in output.items()},
                "selection": {"splits": list(selected_splits), "attacks": sorted(selected_attacks),
                              "skip_texture": args.skip_texture,
                              "shard_index": args.shard_index,
                              "shard_count": args.shard_count},
                "dimensions": {branch: int(output["train"][0]["values"][branch].shape[-1]) for branch in BRANCHES},
                "patch_shape": list(output["train"][0]["values"]["patches"].shape)}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", **metadata}))


if __name__ == "__main__":
    main()
