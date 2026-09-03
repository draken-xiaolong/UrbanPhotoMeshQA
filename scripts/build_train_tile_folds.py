#!/usr/bin/env python3
"""Build deterministic leave-one-Train-tile-out feature folds without formal Val/Test/Blind."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def subset(values: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    count = len(values["asset_ids"])
    return {name: value[indices] if value.ndim > 0 and len(value) == count else value
            for name, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.feature_dir / "features_train.npz") as stored:
        values = {name: stored[name].copy() for name in stored.files}
    sheets = values["sheets"].astype(str)
    folds = []
    for sheet in sorted(set(sheets)):
        validation = np.flatnonzero(sheets == sheet)
        training = np.flatnonzero(sheets != sheet)
        name = sheet.lower().replace("-", "_")
        directory = args.output_root / name
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "features_train.npz", **subset(values, training))
        np.savez_compressed(directory / "features_val.npz", **subset(values, validation))
        folds.append({"id": name, "held_out_sheet": sheet,
                      "train_count": int(len(training)), "val_count": int(len(validation)),
                      "train_sheets": sorted(set(sheets[training]))})
    metadata = {"schema_version": 1, "status": "COMPLETE", "source_split": "train",
                "formal_val_loaded": False, "test_blind_loaded": False, "folds": folds}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
