#!/usr/bin/env python3
"""Compose formal Iteration 2 pseudo-MOS v3 from v2 and perceptual teachers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


GEOMETRY = {"geometry_hole", "geometry_noise_spike", "mesh_simplification_qem"}
LEVEL_ORDER = {"clean": -1, "light": 0, "medium": 1, "heavy": 2}


def soft_min(values: list[float], temperature: float) -> float:
    array = np.asarray(values, np.float64)
    return float(np.clip(-temperature * np.log(np.mean(np.exp(-array / temperature))), 0.0, 1.0))


def reciprocal(value: float) -> float:
    return float(1.0 / (1.0 - np.log(max(float(value), 1e-8))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--objective-v2-dir", type=Path, required=True)
    parser.add_argument("--perceptual-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))["records"]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_by_split = {split: [row for row in manifest if row["split"] == split]
                         for split in ("train", "val", "test", "blind")}
    all_rows = {}
    for split, records in manifest_by_split.items():
        with np.load(args.objective_v2_dir / f"objective_targets_v2_{split}.npz") as v2, \
             np.load(args.perceptual_dir / f"perceptual_teacher_{split}.npz") as teacher:
            v2_keys = list(zip(v2["asset_ids"].astype(str), v2["attacks"].astype(str), v2["levels"].astype(str)))
            teacher_keys = list(zip(teacher["asset_ids"].astype(str), teacher["attacks"].astype(str), teacher["levels"].astype(str)))
            v2_map = {item: i for i, item in enumerate(v2_keys)}
            teacher_map = {item: i for i, item in enumerate(teacher_keys)}
            teacher_names = teacher["metric_names"].astype(str).tolist()
            for record in records:
                item = (record["asset_id"], record["attack"], record["level"])
                i, j = v2_map[item], teacher_map[item]
                metrics = {name: float(teacher["metrics"][j, index])
                           for index, name in enumerate(teacher_names)}
                burden = sum(setting["weight"] * metrics[name] / setting["scale"]
                             for name, setting in config["perceptual_burden"].items())
                perceptual = 1.0 / (1.0 + burden)
                if record["attack"] == "clean":
                    geometry = texture = overall = perceptual = 1.0
                elif record["attack"] in GEOMETRY:
                    base = reciprocal(float(v2["geometry_quality"][i]))
                    geometry = soft_min([base, perceptual], config["component_fusion"]["temperature"])
                    texture = 1.0
                    soft = soft_min([geometry, texture], config["overall_candidates"]["soft_min"]["temperature"])
                    overall = 0.6 * soft + 0.4 * geometry * texture
                else:
                    geometry = 1.0
                    base = reciprocal(float(v2["texture_quality"][i]))
                    texture = soft_min([base, perceptual], config["component_fusion"]["temperature"])
                    soft = soft_min([geometry, texture], config["overall_candidates"]["soft_min"]["temperature"])
                    overall = 0.6 * soft + 0.4 * geometry * texture
                all_rows[item] = {"record": record, "geometry": geometry, "texture": texture,
                                  "overall": overall, "perceptual": perceptual, "metrics": metrics}

    groups = defaultdict(list)
    for item, row in all_rows.items():
        if row["record"]["attack"] != "clean":
            groups[(item[0], item[1])].append(row)
    monotonic_before = monotonic_after = 0
    for group in groups.values():
        group.sort(key=lambda row: LEVEL_ORDER[row["record"]["level"]])
        raw = [row["overall"] for row in group]
        monotonic_before += int(all(a >= b for a, b in zip(raw, raw[1:])))
        burden = 0.0
        for row in group:
            burden = max(burden, 1.0 - row["overall"])
            row["overall"] = 1.0 - burden
            if row["record"]["attack"] in GEOMETRY:
                row["geometry"] = min(row["geometry"], row["overall"])
            else:
                row["texture"] = min(row["texture"], row["overall"])
        final = [row["overall"] for row in group]
        monotonic_after += int(all(a >= b for a, b in zip(final, final[1:])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in manifest_by_split.items():
        rows = [all_rows[(r["asset_id"], r["attack"], r["level"])] for r in records]
        overall = np.asarray([r["overall"] for r in rows], np.float32)
        np.savez_compressed(
            args.output_dir / f"objective_targets_{split}.npz",
            asset_ids=np.asarray([r["record"]["asset_id"] for r in rows]),
            attacks=np.asarray([r["record"]["attack"] for r in rows]),
            levels=np.asarray([r["record"]["level"] for r in rows]),
            overall_quality=overall,
            geometry_quality=np.asarray([r["geometry"] for r in rows], np.float32),
            texture_quality=np.asarray([r["texture"] for r in rows], np.float32),
            ordinal_grade=np.clip(np.ceil(overall * 5), 1, 5).astype(np.int64),
            perceptual_quality=np.asarray([r["perceptual"] for r in rows], np.float32),
        )
    metadata = {
        "schema_version": 3, "name": "Machine Pseudo-MOS v3",
        "selection_protocol": "formula/scales frozen without Val/Test/Blind model results",
        "config": config, "records": len(all_rows), "groups": len(groups),
        "monotonic_before": monotonic_before / max(len(groups), 1),
        "monotonic_after": monotonic_after / max(len(groups), 1),
        "counts": {split: len(rows) for split, rows in manifest_by_split.items()},
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
