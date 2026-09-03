#!/usr/bin/env python3
"""Audit objective OQI targets without changing them or reading model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


SPLITS = ("train", "val", "test", "blind")
LEVELS = ("light", "medium", "heavy")
QUALITY = ("geometry_quality", "texture_quality", "overall_quality")


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(rankdata(left, method="average"), rankdata(right, method="average"))[0, 1])


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "minimum": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(values.max()),
        "at_or_below_0p01": float(np.mean(values <= 0.01)),
        "at_or_above_0p99": float(np.mean(values >= 0.99)),
    }


def load_targets(root: Path) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {}
    constants: dict[str, np.ndarray] = {}
    for split in SPLITS:
        path = root / f"objective_targets_v2_{split}.npz"
        with np.load(path) as data:
            for name in data.files:
                if name.endswith("_metric_names"):
                    current = data[name].copy()
                    if name in constants and not np.array_equal(constants[name], current):
                        raise ValueError(f"Metric names differ across splits: {name}")
                    constants[name] = current
                else:
                    chunks.setdefault(name, []).append(data[name].copy())
            chunks.setdefault("splits", []).append(np.repeat(split, len(data["asset_ids"])))
    merged = {name: np.concatenate(values) for name, values in chunks.items()}
    merged.update(constants)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    values = load_targets(args.target_dir)
    attacks = values["attacks"].astype(str)
    levels = values["levels"].astype(str)
    noops = values["objective_noop"].astype(bool)
    report: dict = {
        "schema_version": 1,
        "status": "PASSED",
        "purpose": "read-only audit of OQI v2 targets before pseudo-MOS v3 design",
        "records": int(len(attacks)),
        "splits": {split: int(np.sum(values["splits"] == split)) for split in SPLITS},
        "objective_noop_records": int(noops.sum()),
        "quality_distribution": {},
        "per_attack_level": {},
        "component_bottleneck": {},
        "metric_srcc_with_component_quality": {},
        "flags": [],
    }
    for name in QUALITY:
        report["quality_distribution"][name] = describe(values[name])

    for attack in sorted(set(attacks) - {"clean"}):
        attack_mask = (attacks == attack) & ~noops
        level_report = {}
        for level in LEVELS:
            mask = attack_mask & (levels == level)
            level_report[level] = {name: describe(values[name][mask]) for name in QUALITY}
        report["per_attack_level"][attack] = level_report
        attacked_quality = values["overall_quality"][attack_mask]
        if len(attacked_quality) and (np.mean(attacked_quality <= 0.01) > 0.20 or
                                     np.mean(attacked_quality >= 0.99) > 0.20):
            report["flags"].append({
                "type": "attack_score_saturation",
                "attack": attack,
                "low_fraction": float(np.mean(attacked_quality <= 0.01)),
                "high_fraction": float(np.mean(attacked_quality >= 0.99)),
            })

    geometry_components = ("geometry_fidelity", "completeness", "topology_health")
    geometry_attacked = np.isin(attacks, ["geometry_hole", "mesh_simplification_qem",
                                           "geometry_noise_spike"])
    component_stack = np.stack([values[name] for name in geometry_components], axis=1)
    winners = np.argmin(component_stack[geometry_attacked], axis=1)
    report["component_bottleneck"] = {
        name: float(np.mean(winners == index)) for index, name in enumerate(geometry_components)
    }

    groups = (
        ("geometry", values["geometry_metrics"], values["geometry_metric_names"].astype(str),
         values["geometry_quality"]),
        ("texture", values["texture_metrics"], values["texture_metric_names"].astype(str),
         values["texture_quality"]),
    )
    for group, metrics, names, quality in groups:
        correlations = {}
        for index, name in enumerate(names):
            valid = np.isfinite(metrics[:, index]) & np.isfinite(quality)
            value = correlation(metrics[valid, index], quality[valid])
            correlations[str(name)] = value
        report["metric_srcc_with_component_quality"][group] = correlations

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# OQI v2机器真值审计", "",
        f"记录数：{report['records']}；objective no-op：{report['objective_noop_records']}。", "",
        "## 全局分布", "",
        "| 分数 | 均值 | 中位数 | ≤0.01 | ≥0.99 |", "|---|---:|---:|---:|---:|",
    ]
    for name, item in report["quality_distribution"].items():
        lines.append(f"| {name} | {item['mean']:.3f} | {item['median']:.3f} | "
                     f"{item['at_or_below_0p01']:.1%} | {item['at_or_above_0p99']:.1%} |")
    lines += ["", "## 各攻击Overall均值", "",
              "| 攻击 | light | medium | heavy |", "|---|---:|---:|---:|"]
    for attack, item in report["per_attack_level"].items():
        lines.append("| " + attack + " | " + " | ".join(
            f"{item[level]['overall_quality']['mean']:.3f}" for level in LEVELS) + " |")
    lines += ["", "## 几何分数瓶颈占比", ""]
    for name, fraction in report["component_bottleneck"].items():
        lines.append(f"- {name}: {fraction:.1%}")
    lines += ["", "## 自动告警", ""]
    if report["flags"]:
        for flag in report["flags"]:
            lines.append(f"- {flag['attack']}分数饱和：低端{flag['low_fraction']:.1%}，高端{flag['high_fraction']:.1%}。")
    else:
        lines.append("- 未触发20%饱和阈值。")
    (args.output_dir / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "records": report["records"],
                      "flags": report["flags"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
