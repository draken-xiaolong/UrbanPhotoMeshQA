#!/usr/bin/env python3
"""Extract foreground-aware spatial texture tokens directly from glTF packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.integrity import asset_digest, extractor_signature, sha256_file
from urbanphotomeshqa.texture import STANDARD_DIRECTIONS, render_textured_view_with_masks
from urbanphotomeshqa.texture_features import (
    ASSET_STAT_NAMES,
    VIEW_STAT_NAMES,
    SpatialImageEncoder,
    texture_asset_statistics,
    texture_quality_statistics,
)


TEXTURE_ATTACKS = ("clean", "texture_detail_loss", "texture_region_missing",
                   "texture_misalignment")
ARRAYS = ("tokens", "token_mask", "view_stats", "asset_stats")


def output_path(root: Path, row: dict) -> Path:
    return (root / row["split"] / row["sheet"] / row["asset_id"] / row["attack"]
            / f"{row['level']}.npz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test", "blind"),
                        default=("train", "val"))
    parser.add_argument("--attacks", nargs="+", default=TEXTURE_ATTACKS)
    parser.add_argument("--render-size", type=int, default=224)
    parser.add_argument("--uv-raster-size", type=int, default=512)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")

    repository = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), repository / "src/urbanphotomeshqa/texture.py",
               repository / "src/urbanphotomeshqa/texture_features.py",
               repository / "src/urbanphotomeshqa/gltf.py"]
    signature_payload = {
        "schema_version": 1, "render_size": args.render_size,
        "directions": STANDARD_DIRECTIONS, "uv_raster_size": args.uv_raster_size,
        "encoder": "torchvision_mobilenet_v3_small_default_feature_map_2x2_masked",
        "view_stat_names": VIEW_STAT_NAMES, "asset_stat_names": ASSET_STAT_NAMES,
        "source_sha256": {str(path.relative_to(repository)): sha256_file(path) for path in sources},
    }
    signature = extractor_signature(signature_payload)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits, attacks = set(args.splits), set(args.attacks)
    records = [row for row in manifest["records"]
               if row["split"] in splits and row["attack"] in attacks]
    records = records[args.shard_index::args.shard_count]
    if args.limit is not None:
        records = records[:args.limit]
    encoder = SpatialImageEncoder(device)
    audit, errors = [], []
    for index, row in enumerate(records):
        gltf = Path(row["gltf_path"])
        digest, dependencies = asset_digest(gltf)
        target = output_path(args.output_root, row)
        if target.is_file() and not args.no_resume:
            try:
                with np.load(target) as stored:
                    metadata = json.loads(str(stored["metadata_json"].item()))
                    valid = (metadata["asset_digest"] == digest
                             and metadata["extractor_signature"] == signature
                             and all(name in stored.files for name in ARRAYS)
                             and all(np.isfinite(stored[name]).all() for name in ("tokens", "view_stats", "asset_stats")))
                if valid:
                    audit.append({**metadata, "feature_path": str(target), "reused": True})
                    print(f"reuse {index + 1}/{len(records)} {row['asset_id']} {row['attack']} {row['level']}", flush=True)
                    continue
            except Exception:
                pass
        try:
            asset = GltfReader(gltf).load_mesh(include_texture=True)
            rendered = [render_textured_view_with_masks(asset, direction, args.render_size)
                        for direction in STANDARD_DIRECTIONS]
            views = np.stack([item[0] for item in rendered])
            foreground = np.stack([item[1] for item in rendered])
            textured = np.stack([item[2] for item in rendered])
            encoded = encoder(views, textured)
            arrays = {**encoded,
                      "view_stats": texture_quality_statistics(views, foreground, textured),
                      "asset_stats": texture_asset_statistics(asset, args.uv_raster_size)}
            if not all(np.isfinite(arrays[name]).all() for name in ("tokens", "view_stats", "asset_stats")):
                raise ValueError("non-finite output")
            metadata = {"schema_version": 1, "asset_id": row["asset_id"], "sheet": row["sheet"],
                        "split": row["split"], "attack": row["attack"], "level": row["level"],
                        "gltf_path": str(gltf), "asset_digest": digest,
                        "extractor_signature": signature, "dependencies": dependencies,
                        "shapes": {name: list(value.shape) for name, value in arrays.items()}}
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **arrays,
                                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
            with np.load(target) as stored:
                if not all(np.array_equal(arrays[name], stored[name]) for name in ARRAYS):
                    raise ValueError("NPZ roundtrip mismatch")
            audit.append({**metadata, "feature_path": str(target), "reused": False})
            print(f"ok {index + 1}/{len(records)} {row['asset_id']} {row['attack']} {row['level']}", flush=True)
        except Exception as error:
            errors.append({"asset_id": row["asset_id"], "attack": row["attack"],
                           "level": row["level"], "error": repr(error)})
            print(f"ERROR {row['asset_id']} {row['attack']} {row['level']} {error!r}", flush=True)
    report = {"schema_version": 1, "passed": not errors, "signature": signature,
              "signature_payload": signature_payload, "selection": {
                  "splits": list(args.splits), "attacks": list(args.attacks),
                  "shard_index": args.shard_index, "shard_count": args.shard_count,
              }, "records": audit, "errors": errors}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "records": len(audit), "errors": len(errors),
                      "signature": signature}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
