#!/usr/bin/env python3
"""Deterministic frozen-feature Ridge baseline selected using Val only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from train_real_gltf_quality import correlation, rankdata


BRANCHES = ("point", "mesh", "morphology", "texture")


def metrics(prediction, truth):
    return {"mae": float(np.mean(np.abs(prediction - truth))),
            "plcc": correlation(prediction, truth),
            "srcc": correlation(rankdata(prediction), rankdata(truth))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--objective-target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1, 10, 100, 1000, 10000])
    args = parser.parse_args()
    raw, targets = {}, {}
    for split in ("train", "val", "test", "blind"):
        with np.load(args.feature_dir / f"features_{split}.npz") as values:
            raw[split] = np.concatenate([values[name].astype(np.float64) for name in BRANCHES], 1)
        with np.load(args.objective_target_dir / f"objective_targets_{split}.npz") as values:
            targets[split] = np.column_stack([values["overall_quality"], values["geometry_quality"],
                                              values["texture_quality"]]).astype(np.float64)
    mean, std = raw["train"].mean(0), np.maximum(raw["train"].std(0), 1e-5)
    data = {split: (values - mean) / std for split, values in raw.items()}
    candidates = []
    for alpha in args.alphas:
        model = Ridge(alpha=alpha).fit(data["train"], targets["train"])
        prediction = np.clip(model.predict(data["val"]), 0, 1)
        result = metrics(prediction[:, 0], targets["val"][:, 0])
        candidates.append({"alpha": alpha, "val_overall": result})
    selected = max(candidates, key=lambda row: (row["val_overall"]["srcc"],
                                                 -row["val_overall"]["mae"]))
    model = Ridge(alpha=selected["alpha"]).fit(data["train"], targets["train"])
    results = {}
    for split in ("val", "test", "blind"):
        prediction = np.clip(model.predict(data[split]), 0, 1)
        results[split] = {name: metrics(prediction[:, index], targets[split][:, index])
                          for index, name in enumerate(("overall", "geometry", "texture"))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "ridge_model.npz", coefficient=model.coef_, intercept=model.intercept_,
                        mean=mean, std=std, alpha=np.asarray(selected["alpha"]), branches=np.asarray(BRANCHES))
    report = {"schema_version": 1, "status": "COMPLETE", "deterministic": True,
              "selection": "alpha chosen by Val OQI SRCC, then MAE; Test/Blind locked",
              "selected_alpha": selected["alpha"], "candidates": candidates, "results": results}
    (args.output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
