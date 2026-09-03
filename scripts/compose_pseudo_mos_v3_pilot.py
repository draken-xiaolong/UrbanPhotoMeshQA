#!/usr/bin/env python3
"""Compose transparent pseudo-MOS v3 candidates for the fixed 18-case Pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LEVELS = ("light", "medium", "heavy")
GEOMETRY = {"geometry_hole", "geometry_noise_spike", "mesh_simplification_qem"}


def reciprocal_quality(v2_quality: float) -> float:
    burden = -np.log(max(float(v2_quality), 1e-8))
    return float(1.0 / (1.0 + burden))


def soft_min(values: list[float], temperature: float) -> float:
    array = np.asarray(values, np.float64)
    result = -temperature * np.log(np.mean(np.exp(-array / temperature)))
    return float(np.clip(result, 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    teacher = json.loads(args.teacher.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    spec = config["perceptual_burden"]
    rows = []
    for case in teacher["cases"]:
        for level in LEVELS:
            source = case["levels"][level]
            perceptual_burden = sum(
                setting["weight"] * source[name] / setting["scale"]
                for name, setting in spec.items()
            )
            perceptual = 1.0 / (1.0 + perceptual_burden)
            if case["attack"] in GEOMETRY:
                base = reciprocal_quality(source["geometry_quality"])
                geometry = soft_min([base, perceptual], config["component_fusion"]["temperature"])
                texture = 1.0
            else:
                geometry = 1.0
                base = reciprocal_quality(source["texture_quality"])
                texture = soft_min([base, perceptual], config["component_fusion"]["temperature"])
            hard = min(geometry, texture)
            soft = soft_min([geometry, texture], config["overall_candidates"]["soft_min"]["temperature"])
            interaction = 0.6 * soft + 0.4 * geometry * texture
            rows.append({
                "asset_id": case["asset_id"], "attack": case["attack"], "level": level,
                "oqi_v2": source["overall_quality"], "base_reciprocal": base,
                "perceptual_quality": perceptual, "geometry_quality_v3": geometry,
                "texture_quality_v3": texture, "overall_hard_min": hard,
                "overall_soft_min": soft, "overall_bounded_interaction": interaction,
            })
    monotonic = {}
    for attack in sorted({row["attack"] for row in rows}):
        selected = [row for row in rows if row["attack"] == attack]
        values = [row["overall_bounded_interaction"] for row in selected]
        monotonic[attack] = bool(values[0] >= values[1] >= values[2])
    report = {
        "schema_version": 1, "status": "PASSED" if all(monotonic.values()) else "FAILED",
        "protocol": config["selection_protocol"], "teacher": teacher["teacher"],
        "monotonic": monotonic, "records": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pseudo_mos_v3_pilot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# Pseudo-MOS v3 Pilot", "", "| 攻击 | 等级 | OQI v2 | 感知质量 | v3几何 | v3纹理 | v3综合 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['attack']} | {row['level']} | {row['oqi_v2']:.3f} | "
                     f"{row['perceptual_quality']:.3f} | {row['geometry_quality_v3']:.3f} | "
                     f"{row['texture_quality_v3']:.3f} | {row['overall_bounded_interaction']:.3f} |")
    lines += ["", f"三级单调：{sum(monotonic.values())}/{len(monotonic)}类通过。", "",
              "该结果仅用于验证公式形态；正式尺度必须在扩容后的Train上冻结。"]
    (args.output_dir / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "monotonic": monotonic}, ensure_ascii=False))
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
