#!/usr/bin/env python3
"""Summarize leave-one-Train-tile-out quality results without formal Val/Test/Blind."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-metadata", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.fold_metadata.read_text(encoding="utf-8"))
    rows = []
    for fold in metadata["folds"]:
        result = json.loads((args.run_root / fold["id"] / "results.json").read_text(encoding="utf-8"))
        if result["protocol"].get("test_blind_loaded") is not False:
            raise ValueError(f"Locked split protocol violation: {fold['id']}")
        metrics = result["variants"]["shared"]["val"]
        rows.append({"fold": fold["id"], "held_out_sheet": fold["held_out_sheet"],
                     "count": fold["val_count"], "overall": metrics["overall"],
                     "geometry": metrics["geometry"], "texture": metrics["texture"]})
    aggregate = {}
    for target in ("overall", "geometry", "texture"):
        aggregate[target] = {}
        for metric in ("mae", "plcc", "srcc"):
            values = np.asarray([row[target][metric] for row in rows], np.float64)
            aggregate[target][metric] = {"mean": float(values.mean()), "worst": (
                float(values.max()) if metric == "mae" else float(values.min())),
                "values": values.tolist()}
    payload = {"schema_version": 1, "status": "COMPLETE", "seed": 2026,
               "protocol": {"source_split": "train", "formal_val_loaded": False,
                            "test_blind_loaded": False}, "folds": rows, "aggregate": aggregate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
