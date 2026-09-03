#!/usr/bin/env python3
"""Create content-addressed raw NPZ caches strictly from exported glTF packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from urbanphotomeshqa.gltf import GltfReader, sample_surface
from urbanphotomeshqa.integrity import asset_digest, extractor_signature
from urbanphotomeshqa.mesh_attacks import mesh_face_graph
from urbanphotomeshqa.patches import patch_descriptors
from urbanphotomeshqa.render_features import render_views


ARRAYS = ("points", "face_features", "neighbors", "topology", "patches", "patch_mask", "render_views")


def extract(gltf: Path, settings: dict) -> tuple[dict[str, np.ndarray], dict]:
    asset = GltfReader(gltf).load_mesh(include_texture=True)
    points, sampling = sample_surface(asset, settings["points"], settings["seed"])
    face_features, neighbors, topology = mesh_face_graph(asset)
    patches, patch_mask = patch_descriptors(asset, settings["patches"], settings["patch_neighbors"])
    views = np.stack(render_views(asset, settings["render_size"])).astype(np.uint8)
    arrays = {
        "points": points.astype(np.float32), "face_features": face_features.astype(np.float32),
        "neighbors": neighbors.astype(np.int64), "topology": topology.astype(np.float32),
        "patches": patches.astype(np.float32), "patch_mask": patch_mask,
        "render_views": views,
    }
    return arrays, {"sampling": sampling, "vertex_count": len(asset.vertices), "face_count": len(asset.faces)}


def equivalent(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict:
    checks = {}
    for name in ARRAYS:
        a, b = left[name], right[name]
        if np.issubdtype(a.dtype, np.floating):
            ok = a.shape == b.shape and np.allclose(a, b, rtol=1e-6, atol=1e-7)
            maximum = float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else 0.0
        else:
            ok = np.array_equal(a, b)
            maximum = 0.0
        checks[name] = {"ok": bool(ok), "shape": list(a.shape), "max_abs": maximum}
    return checks


def output_path(root: Path, record: dict) -> Path:
    attack, level = record.get("attack", "clean"), record.get("level", "clean")
    return root / record["sheet"] / "BUILDING" / record["asset_id"] / attack / level / "features.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--patches", type=int, default=16)
    parser.add_argument("--patch-neighbors", type=int, default=32)
    parser.add_argument("--render-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--verify-reextract", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--attack-filter", nargs="*")
    args = parser.parse_args()
    settings = {"schema_version": 1, "points": args.points, "patches": args.patches,
                "patch_neighbors": args.patch_neighbors, "render_size": args.render_size,
                "render_directions": 6, "seed": args.seed}
    signature = extractor_signature(settings)
    records = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    if args.attack_filter:
        allowed = set(args.attack_filter)
        records = [record for record in records if record.get("attack", "clean") in allowed]
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    records = records[args.shard_index::args.shard_count]
    audit, errors = [], []
    for index, record in enumerate(records):
        gltf = Path(record["gltf_path"])
        digest, dependencies = asset_digest(gltf)
        target = output_path(args.output_root, record)
        if target.is_file() and not args.no_resume:
            try:
                with np.load(target) as stored:
                    cached_metadata = json.loads(str(stored["metadata_json"].item()))
                    stored_arrays = {name: stored[name].copy() for name in ARRAYS}
                if (cached_metadata.get("asset_digest") == digest
                        and cached_metadata.get("extractor_signature") == signature):
                    audit.append({**cached_metadata, "cache_path": str(target), "reused": True,
                                  "reextract": None, "roundtrip": equivalent(stored_arrays, stored_arrays)})
                    print(f"reuse {index + 1}/{len(records)} {record['asset_id']} {record.get('attack')} {record.get('level')}", flush=True)
                    continue
            except Exception:
                pass
        arrays, details = extract(gltf, settings)
        checks = None
        if args.verify_reextract:
            second, _ = extract(gltf, settings)
            checks = equivalent(arrays, second)
            if not all(value["ok"] for value in checks.values()):
                errors.append(str(gltf))
        metadata = {"schema_version": 1, "asset_id": record["asset_id"],
                    "sheet": record["sheet"], "split": record["split"],
                    "attack": record.get("attack", "clean"), "level": record.get("level", "clean"),
                    "gltf_path": str(gltf), "asset_digest": digest,
                    "extractor_signature": signature, "extractor_settings": settings,
                    "dependencies": dependencies, **details}
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays, metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
        with np.load(target) as stored:
            stored_arrays = {name: stored[name].copy() for name in ARRAYS}
            roundtrip = equivalent(arrays, stored_arrays)
        if not all(value["ok"] for value in roundtrip.values()):
            errors.append(f"NPZ roundtrip: {target}")
        audit.append({**metadata, "cache_path": str(target), "reused": False,
                      "reextract": checks, "roundtrip": roundtrip})
        print(f"ok {index + 1}/{len(records)} {record['asset_id']} {record.get('attack')} {record.get('level')}", flush=True)
    report = {"schema_version": 1, "passed": not errors, "errors": errors,
              "extractor_signature": signature, "records": audit}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "records": len(audit), "errors": errors}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
