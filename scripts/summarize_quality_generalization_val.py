#!/usr/bin/env python3
"""Select exactly one generalization candidate using validation metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=root / "configs/quality_generalization_minimal_seed2026.json")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    variant = config["common_training"]["variant"]
    rows, provenance = [], None
    for run in config["runs"]:
        path = args.run_root / run["id"] / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("seed") != config["seed"]:
            raise ValueError(f"Seed mismatch in {path}")
        current_provenance = payload["protocol"].get("dataset_provenance")
        if not current_provenance or not current_provenance.get("formal"):
            raise ValueError(f"Formal provenance missing in {path}")
        signature = current_provenance["ordered_sample_sha256"]
        if provenance is None:
            provenance = signature
        elif signature != provenance:
            raise ValueError("Candidate dataset provenance differs")
        metrics = payload["variants"][variant]["results"]
        if set(metrics) != {"val"}:
            raise ValueError(f"Locked split was evaluated during candidate selection: {path}")
        val = metrics["val"]
        tile_srcc = [item["srcc"] for item in val.get("per_tile", {}).values()]
        rows.append({
            "id": run["id"],
            "checkpoint": f"{run['id']}/{variant}.pt",
            "best_epoch": payload["variants"][variant]["best_epoch"],
            "val_oqi_srcc": val["overall"]["srcc"],
            "val_oqi_plcc": val["overall"]["plcc"],
            "val_oqi_mae": val["overall"]["mae"],
            "val_geometry_srcc": val["geometry"]["srcc"],
            "val_texture_srcc": val["texture"]["srcc"],
            "val_worst_tile_srcc": min(tile_srcc) if tile_srcc else None,
        })

    release = config.get("release_val_reference")
    if release:
        baseline = {
            "id": release["id"], "checkpoint": release["checkpoint"],
            "val_oqi_srcc": release["overall"]["srcc"],
            "val_oqi_plcc": release["overall"]["plcc"],
            "val_oqi_mae": release["overall"]["mae"],
            "val_geometry_srcc": release["geometry"]["srcc"],
            "val_texture_srcc": release["texture"]["srcc"],
        }
    else:
        baseline = next((row for row in rows if row["id"].startswith("B0_")), None)
        if baseline is None:
            raise ValueError("B0 baseline is required by the promotion gate")
    gate = config["promotion_gate"]
    for row in rows:
        row["gain_over_reference"] = row["val_oqi_srcc"] - baseline["val_oqi_srcc"]
        row["geometry_drop_from_reference"] = baseline["val_geometry_srcc"] - row["val_geometry_srcc"]
        row["texture_drop_from_reference"] = baseline["val_texture_srcc"] - row["val_texture_srcc"]
        row["passes_promotion_gate"] = (
            row["id"] != baseline["id"]
            and row["gain_over_reference"] >= gate["minimum_val_oqi_srcc_gain_over_reference"]
            and row["geometry_drop_from_reference"] <= gate["maximum_allowed_val_geometry_srcc_drop"]
            and row["texture_drop_from_reference"] <= gate["maximum_allowed_val_texture_srcc_drop"]
        )

    eligible = [row for row in rows if row["passes_promotion_gate"]]
    selected = max(eligible, key=lambda row: (
        row["val_oqi_srcc"], -row["val_oqi_mae"], row["val_oqi_plcc"]
    )) if eligible else baseline
    report = {
        "schema_version": 1,
        "status": "VAL_SELECTION_COMPLETE",
        "seed": config["seed"],
        "protocol": {
            "selection_data": "Val only",
            "test_blind_evaluated": False,
            "ordered_sample_sha256": provenance,
            "promotion_gate": gate,
        },
        "selected_id": selected["id"],
        "selected_checkpoint": selected["checkpoint"],
        "promoted_over_baseline": selected["id"] != baseline["id"],
        "promotion_reference": baseline,
        "candidates": rows,
        "next_step": "Freeze this selection, then run calibration/evaluation exactly once on Test and Blind.",
    }
    output = args.output or args.run_root / "val_selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
