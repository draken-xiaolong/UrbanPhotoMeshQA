#!/usr/bin/env python3
"""Plan or execute the seed-2026 frozen-Base generalization screen on GPU."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=root / "configs/quality_generalization_minimal_seed2026.json")
    parser.add_argument("--feature-dir", type=Path, default=root /
                        "artifacts/quality/final/frozen_features_real_gltf_formal_seed2026_v2")
    parser.add_argument("--objective-target-dir", type=Path, default=root /
                        "artifacts/quality/final/objective_targets_real_gltf_formal_seed2026_v2")
    parser.add_argument("--manifest", type=Path, default=root /
                        "artifacts/manifests/quality_dataset_formal_seed2026.json")
    parser.add_argument("--output-root", type=Path, default=root /
                        "artifacts/quality/ablations/generalization_minimal_seed2026_v1")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true",
                        help="Execute commands; without this flag only print the plan")
    parser.add_argument("--resume", action="store_true",
                        help="Skip candidates that already have a COMPLETE results.json")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def display(command: list[str]) -> str:
    return shlex.join(command)


def run(command: list[str], *, env=None) -> None:
    print(f"+ {display(command)}", flush=True)
    subprocess.run(command, check=True, env=env)


def completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "COMPLETE"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("seed") != 2026:
        raise ValueError("This formal screen is fixed to the single seed 2026")
    common = config["common_training"]
    if common.get("evaluate_locked") is not False:
        raise ValueError("Candidate screen must keep Test/Blind locked")
    for candidate in config["runs"]:
        if "--evaluate-locked" in candidate.get("cli_args", []):
            raise ValueError(f"Locked evaluation forbidden in {candidate['id']}")

    audit = [
        sys.executable, str(root / "scripts/audit_gpu_quality_environment.py"),
        "--manifest", str(args.manifest),
        "--feature-dir", str(args.feature_dir),
        "--objective-target-dir", str(args.objective_target_dir),
        "--require-cuda", "--output", str(args.output_root / "gpu_environment_audit.json"),
    ]
    if args.data_root:
        audit.extend(["--data-root", str(args.data_root)])

    commands = []
    for candidate in config["runs"]:
        output_dir = args.output_root / candidate["id"]
        commands.append([
            sys.executable, str(root / "scripts/train_real_gltf_quality.py"),
            "--feature-dir", str(args.feature_dir),
            "--objective-target-dir", str(args.objective_target_dir),
            "--dataset-manifest", str(args.manifest), "--require-formal",
            "--output-dir", str(output_dir),
            "--variants", common["variant"],
            "--epochs", str(common["epochs"]),
            "--batch-size", str(common["batch_size"]),
            "--lr", str(common["learning_rate"]),
            "--seed", str(config["seed"]), "--device", args.device,
            *candidate.get("cli_args", []),
        ])
    summary = [
        sys.executable, str(root / "scripts/summarize_quality_generalization_val.py"),
        "--config", str(args.config), "--run-root", str(args.output_root),
    ]

    print("GPU audit:")
    print(display(audit))
    print("\nVal-only candidates (Test/Blind NPZ files are not loaded):")
    for command in commands:
        print(display(command))
    print("\nVal selection:")
    print(display(summary))
    if not args.execute:
        print("\nPLAN ONLY: add --execute to run.")
        return

    if not args.allow_dirty:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if status:
            raise RuntimeError("Refusing to start from a dirty Git worktree; commit/push or use --allow-dirty")

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(config["seed"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    run(audit, env=environment)
    for candidate, command in zip(config["runs"], commands):
        result_path = args.output_root / candidate["id"] / "results.json"
        if result_path.exists():
            if args.resume and completed(result_path):
                print(f"SKIP complete candidate: {candidate['id']}", flush=True)
                continue
            raise FileExistsError(
                f"Refusing to overwrite candidate output: {result_path}; use a new output root"
            )
        run(command, env=environment)
    run(summary, env=environment)


if __name__ == "__main__":
    main()
