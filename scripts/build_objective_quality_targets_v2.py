#!/usr/bin/env python3
"""Compose audited geometry and texture metrics into interpretable OQI v2 targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GEOMETRY_ATTACKS = {
    "geometry_hole",
    "mesh_simplification_qem",
    "geometry_noise_spike",
}
TEXTURE_ATTACKS = {
    "texture_detail_loss",
    "texture_region_missing",
    "texture_misalignment",
}
LEVEL_ORDER = {"light": 0, "medium": 1, "heavy": 2}


def load_metric_targets(root: Path, prefix: str):
    rows = {}
    names = None
    extras = {}
    for split in ("train", "val", "test", "blind"):
        with np.load(root / f"{prefix}_{split}.npz") as values:
            current = values["metric_names"].astype(str).tolist()
            if names is None:
                names = current
            elif names != current:
                raise ValueError(f"Metric names differ in {root}/{split}")
            keys = list(zip(
                values["asset_ids"].astype(str),
                values["attacks"].astype(str),
                values["levels"].astype(str),
            ))
            for index, key in enumerate(keys):
                rows[key] = values["metrics"][index].astype(np.float64)
                if "objective_noop" in values:
                    extras[key] = bool(values["objective_noop"][index])
    return names, rows, extras


def component_quality(metrics, names, specification):
    index = {name: position for position, name in enumerate(names)}
    burden = 0.0
    for name, settings in specification.items():
        if name not in index:
            raise ValueError(f"Required metric is missing: {name}")
        burden += (
            float(settings["weight"])
            * max(float(metrics[index[name]]), 0.0)
            / float(settings["scale"])
        )
    return float(np.exp(-burden))


def key(row):
    return row["asset_id"], row["attack"], row["level"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--geometry-target-dir", type=Path, required=True)
    parser.add_argument("--texture-target-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))["records"]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    geometry_names, geometry_metrics, _ = load_metric_targets(
        args.geometry_target_dir, "geometry_targets_v2"
    )
    texture_names, texture_metrics, texture_noops = load_metric_targets(
        args.texture_target_dir, "texture_targets_v2"
    )

    rows = []
    for record in manifest:
        record_key = key(record)
        attack = record["attack"]
        clean_key = (record["asset_id"], "clean", "clean")
        geometry_vector = geometry_metrics.get(record_key, geometry_metrics[clean_key])
        texture_vector = texture_metrics.get(record_key, texture_metrics[clean_key])
        geometry_fidelity = component_quality(
            geometry_vector, geometry_names, config["geometry_fidelity"]
        )
        completeness = component_quality(
            geometry_vector, geometry_names, config["completeness"]
        )
        topology_health = component_quality(
            geometry_vector, geometry_names, config["topology_health"]
        )
        texture_fidelity = component_quality(
            texture_vector, texture_names, config["texture_fidelity"]
        )
        geometry_quality_raw = min(geometry_fidelity, completeness, topology_health)
        objective_noop = bool(
            attack in TEXTURE_ATTACKS and texture_noops.get(record_key, False)
        )
        if objective_noop:
            texture_fidelity = 1.0
        overall_raw = min(geometry_quality_raw, texture_fidelity)
        rows.append({
            "record": record,
            "geometry_metrics": geometry_vector,
            "texture_metrics": texture_vector,
            "geometry_fidelity": geometry_fidelity,
            "completeness": completeness,
            "topology_health": topology_health,
            "texture_fidelity_raw": texture_fidelity,
            "geometry_quality_raw": geometry_quality_raw,
            "overall_quality_raw": overall_raw,
            "geometry_quality": geometry_quality_raw,
            "texture_quality": texture_fidelity,
            "overall_quality": overall_raw,
            "objective_noop": objective_noop,
        })

    monotonic_before = {}
    monotonic_after = {}
    for attack in sorted(GEOMETRY_ATTACKS | TEXTURE_ATTACKS):
        groups = {}
        for row in rows:
            if row["record"]["attack"] == attack:
                groups.setdefault(row["record"]["asset_id"], []).append(row)
        before = 0
        after = 0
        considered = 0
        for group in groups.values():
            group.sort(key=lambda row: LEVEL_ORDER[row["record"]["level"]])
            observable = [row for row in group if not row["objective_noop"]]
            raw = [row["overall_quality_raw"] for row in observable]
            before += int(all(left >= right - 1e-12 for left, right in zip(raw, raw[1:])))
            considered += 1
            burden = 0.0
            for row in group:
                if row["objective_noop"]:
                    continue
                burden = max(burden, 1.0 - row["overall_quality_raw"])
                row["overall_quality"] = 1.0 - burden
                if attack in GEOMETRY_ATTACKS:
                    row["geometry_quality"] = row["overall_quality"]
                else:
                    row["texture_quality"] = row["overall_quality"]
            final = [row["overall_quality"] for row in observable]
            after += int(all(left >= right - 1e-12 for left, right in zip(final, final[1:])))
        monotonic_before[attack] = {"passed": before, "total": considered,
                                     "rate": before / max(considered, 1)}
        monotonic_after[attack] = {"passed": after, "total": considered,
                                    "rate": after / max(considered, 1)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test", "blind"):
        selected = [row for row in rows if row["record"]["split"] == split]
        counts[split] = len(selected)
        np.savez_compressed(
            args.output_dir / f"objective_targets_v2_{split}.npz",
            asset_ids=np.asarray([row["record"]["asset_id"] for row in selected]),
            attacks=np.asarray([row["record"]["attack"] for row in selected]),
            levels=np.asarray([row["record"]["level"] for row in selected]),
            geometry_metrics=np.stack([row["geometry_metrics"] for row in selected]).astype(np.float32),
            geometry_metric_names=np.asarray(geometry_names),
            texture_metrics=np.stack([row["texture_metrics"] for row in selected]).astype(np.float32),
            texture_metric_names=np.asarray(texture_names),
            geometry_fidelity=np.asarray([row["geometry_fidelity"] for row in selected], np.float32),
            completeness=np.asarray([row["completeness"] for row in selected], np.float32),
            topology_health=np.asarray([row["topology_health"] for row in selected], np.float32),
            texture_fidelity_raw=np.asarray([row["texture_fidelity_raw"] for row in selected], np.float32),
            geometry_quality_raw=np.asarray([row["geometry_quality_raw"] for row in selected], np.float32),
            overall_quality_raw=np.asarray([row["overall_quality_raw"] for row in selected], np.float32),
            geometry_quality=np.asarray([row["geometry_quality"] for row in selected], np.float32),
            texture_quality=np.asarray([row["texture_quality"] for row in selected], np.float32),
            overall_quality=np.asarray([row["overall_quality"] for row in selected], np.float32),
            objective_noop=np.asarray([row["objective_noop"] for row in selected], bool),
        )

    metadata = {
        "schema_version": 2,
        "supervision": "full-reference objective target generation; deployed inference remains no-reference",
        "configuration": config,
        "counts": counts,
        "monotonicity_before_envelope_observable_records": monotonic_before,
        "monotonicity_after_envelope_observable_records": monotonic_after,
        "objective_noop_records": int(sum(row["objective_noop"] for row in rows)),
        "dataset_manifest": str(args.dataset_manifest),
        "geometry_target_dir": str(args.geometry_target_dir),
        "texture_target_dir": str(args.texture_target_dir),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETE", **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
