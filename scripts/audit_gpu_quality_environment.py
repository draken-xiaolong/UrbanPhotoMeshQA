#!/usr/bin/env python3
"""Fail-fast audit for the formal GPU quality-training environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np


FORMAL_COUNTS = {"train": 1518, "val": 760, "test": 607, "blind": 608}
REQUIRED_FEATURE_ARRAYS = {
    "asset_ids", "attacks", "levels", "attack_index", "severity",
    "point", "mesh", "morphology", "texture", "patches", "patch_mask",
}
REQUIRED_TARGET_ARRAYS = {
    "asset_ids", "attacks", "levels", "overall_quality", "geometry_quality",
    "texture_quality", "patch_quality",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_materialized(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        prefix = stream.read(80)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(f"Git LFS object is not materialized: {path}; run git lfs pull")


def identity(values) -> list[tuple[str, str, str]]:
    return list(zip(values["asset_ids"].astype(str), values["attacks"].astype(str),
                    values["levels"].astype(str)))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=root / "artifacts/manifests/quality_dataset_formal_seed2026.json")
    parser.add_argument("--feature-dir", type=Path, default=root /
                        "artifacts/quality/final/frozen_features_real_gltf_formal_seed2026_v2")
    parser.add_argument("--objective-target-dir", type=Path, default=root /
                        "artifacts/quality/final/objective_targets_real_gltf_formal_seed2026_v2")
    parser.add_argument("--point-checkpoint", type=Path, default=root /
                        "artifacts/pretrained_backbone/formal_invariant_seed2026_v1/best.pt")
    parser.add_argument("--mesh-checkpoint", type=Path, default=root /
                        "artifacts/pretrained_backbone/native_mesh_v1/best.pt")
    parser.add_argument("--release-dir", type=Path, default=root /
                        "artifacts/quality/final/release_seed2026_v1")
    parser.add_argument("--data-root", type=Path,
                        help="Optional server data root containing source, attacks, and raw caches")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.manifest, args.point_checkpoint, args.mesh_checkpoint):
        ensure_materialized(path)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    counts = dict(Counter(str(row["split"]) for row in records))
    if counts != FORMAL_COUNTS:
        raise ValueError(f"Formal split counts mismatch: {counts}")
    exclusions = manifest.get("quality_control", {}).get(
        "excluded_exact_duplicate_severity_packages")
    if exclusions != 3:
        raise ValueError(f"Expected exactly 3 formal exclusions, got {exclusions}")

    expected = {
        split: [(str(row["asset_id"]), str(row["attack"]), str(row["level"]))
                for row in records if row["split"] == split]
        for split in FORMAL_COUNTS
    }
    arrays = {}
    for split, count in FORMAL_COUNTS.items():
        feature_path = args.feature_dir / f"features_{split}.npz"
        target_path = args.objective_target_dir / f"objective_targets_{split}.npz"
        ensure_materialized(feature_path)
        ensure_materialized(target_path)
        with np.load(feature_path) as feature, np.load(target_path) as target:
            missing_features = REQUIRED_FEATURE_ARRAYS.difference(feature.files)
            missing_targets = REQUIRED_TARGET_ARRAYS.difference(target.files)
            if missing_features or missing_targets:
                raise ValueError(
                    f"{split} missing arrays: features={sorted(missing_features)}, "
                    f"targets={sorted(missing_targets)}"
                )
            feature_identity, target_identity = identity(feature), identity(target)
            if feature_identity != expected[split] or target_identity != expected[split]:
                raise ValueError(f"Formal row order mismatch: {split}")
            if len(feature_identity) != count:
                raise ValueError(f"Formal row count mismatch: {split}")
            arrays[split] = {
                "records": count,
                "point_dim": int(feature["point"].shape[1]),
                "mesh_dim": int(feature["mesh"].shape[1]),
                "morphology_dim": int(feature["morphology"].shape[1]),
                "texture_dim": int(feature["texture"].shape[1]),
                "patch_shape": list(feature["patches"].shape[1:]),
            }

    checksum_path = args.release_dir / "SHA256SUMS.txt"
    ensure_materialized(checksum_path)
    checksum_results = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        candidate = args.release_dir / filename.lstrip("* ")
        ensure_materialized(candidate)
        actual = sha256_file(candidate)
        checksum_results[candidate.name] = actual == digest
        if actual != digest:
            raise ValueError(f"Release checksum mismatch: {candidate}")

    data_paths = None
    if args.data_root is not None:
        candidates = {
            "source": args.data_root / "HK3D-Individualised",
            "attacks": args.data_root / "HK3D-Individualised-Attack",
            "raw_cache": args.data_root / "Feature-Cache/Full",
        }
        missing = [str(path) for path in candidates.values() if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Server data directories missing: {missing}")
        data_paths = {name: str(path.resolve()) for name, path in candidates.items()}

    cuda = {"required": args.require_cuda, "available": False}
    try:
        import torch
        cuda.update({"torch": torch.__version__, "available": torch.cuda.is_available()})
        if torch.cuda.is_available():
            cuda.update({"device_count": torch.cuda.device_count(),
                         "device_0": torch.cuda.get_device_name(0)})
    except ImportError:
        cuda["torch"] = None
    if args.require_cuda and not cuda["available"]:
        raise RuntimeError("CUDA GPU is required but torch.cuda.is_available() is false")

    ordered = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
    report = {
        "schema_version": 1,
        "status": "PASS",
        "seed": manifest.get("seed"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "manifest": str(args.manifest.resolve()),
        "ordered_sample_sha256": hashlib.sha256(ordered.encode("utf-8")).hexdigest(),
        "formal_counts": counts,
        "formal_exclusions": exclusions,
        "arrays": arrays,
        "checkpoints": {
            "point": {"path": str(args.point_checkpoint.resolve()),
                      "sha256": sha256_file(args.point_checkpoint)},
            "mesh": {"path": str(args.mesh_checkpoint.resolve()),
                     "sha256": sha256_file(args.mesh_checkpoint)},
        },
        "release_checksums": checksum_results,
        "data_paths": data_paths,
        "cuda": cuda,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
