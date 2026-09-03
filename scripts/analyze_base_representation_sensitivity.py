#!/usr/bin/env python3
"""Measure frozen Point/Mesh feature sensitivity on Train/Val geometry attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPRESENTATIONS = ("identity", "global")
BRANCHES = ("point", "mesh")
LEVELS = ("light", "medium", "heavy")


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float:
    if rank:
        left, right = rankdata(left), rankdata(right)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def load_targets(root: Path, split: str) -> dict[tuple[str, str, str], float]:
    path = root / f"objective_targets_v2_{split}.npz"
    if not path.is_file():
        path = root / f"objective_targets_{split}.npz"
    with np.load(path) as values:
        keys = zip(values["asset_ids"].astype(str), values["attacks"].astype(str),
                   values["levels"].astype(str))
        return {key: float(value) for key, value in zip(keys, values["geometry_quality"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--objective-target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = {}
    for split in ("train", "val"):
        with np.load(args.feature_dir / f"features_{split}.npz") as values:
            data[split] = {name: values[name].copy() for name in values.files}

    statistics = {}
    clean_train = data["train"]["attacks"].astype(str) == "clean"
    for representation in REPRESENTATIONS:
        statistics[representation] = {}
        for branch in BRANCHES:
            values = data["train"][f"{branch}_{representation}"][clean_train].astype(np.float64)
            statistics[representation][branch] = (
                values.mean(axis=0), np.maximum(values.std(axis=0), 1e-6)
            )

    output = {"schema_version": 1, "protocol": {
        "splits": ["train", "val"], "test_blind_loaded": False,
        "reference_pairing": "diagnostic only; never used by inference or model input",
        "standardization": "Train-clean mean/std",
    }, "representations": {}}
    for representation in REPRESENTATIONS:
        output["representations"][representation] = {}
        for split in ("train", "val"):
            values = data[split]
            asset_ids = values["asset_ids"].astype(str)
            attacks = values["attacks"].astype(str)
            levels = values["levels"].astype(str)
            target_lookup = load_targets(args.objective_target_dir, split)
            keys = list(zip(asset_ids, attacks, levels))
            distortion = np.asarray([1.0 - target_lookup[key] for key in keys])
            clean_index = {asset_id: index for index, (asset_id, attack) in enumerate(
                zip(asset_ids, attacks)) if attack == "clean"}
            branch_scores = {}
            for branch in BRANCHES:
                feature = values[f"{branch}_{representation}"].astype(np.float64)
                center, scale = statistics[representation][branch]
                standardized = (feature - center) / scale
                reference = np.stack([standardized[clean_index[asset_id]] for asset_id in asset_ids])
                branch_scores[branch] = np.sqrt(np.mean((standardized - reference) ** 2, axis=1))
            branch_scores["joint"] = np.sqrt(
                0.5 * branch_scores["point"] ** 2 + 0.5 * branch_scores["mesh"] ** 2
            )
            attacked = attacks != "clean"
            split_result = {}
            for branch, score in branch_scores.items():
                result = {
                    "attacked_count": int(attacked.sum()),
                    "distortion_plcc": correlation(score[attacked], distortion[attacked]),
                    "distortion_srcc": correlation(score[attacked], distortion[attacked], rank=True),
                    "per_attack": {}, "per_level": {},
                }
                for attack in sorted(set(attacks[attacked])):
                    mask = attacks == attack
                    result["per_attack"][attack] = {
                        "count": int(mask.sum()), "median_distance": float(np.median(score[mask])),
                        "distortion_srcc": correlation(score[mask], distortion[mask], rank=True),
                    }
                for level in LEVELS:
                    mask = levels == level
                    result["per_level"][level] = {
                        "count": int(mask.sum()), "median_distance": float(np.median(score[mask]))
                    }
                monotonic, total = 0, 0
                for asset_id in sorted(set(asset_ids)):
                    for attack in sorted(set(attacks[attacked])):
                        indices = [np.flatnonzero((asset_ids == asset_id) & (attacks == attack)
                                                 & (levels == level)) for level in LEVELS]
                        if all(len(index) == 1 for index in indices):
                            distances = [score[index[0]] for index in indices]
                            monotonic += int(distances[0] <= distances[1] <= distances[2])
                            total += 1
                result["severity_monotonic"] = {"passed": monotonic, "total": total,
                                                 "rate": monotonic / max(total, 1)}
                split_result[branch] = result
            output["representations"][representation][split] = split_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({representation: output["representations"][representation]["val"]["joint"]
                      for representation in REPRESENTATIONS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
