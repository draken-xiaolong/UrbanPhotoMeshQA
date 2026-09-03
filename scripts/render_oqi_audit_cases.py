#!/usr/bin/env python3
"""Render deterministic representative OQI cases for visual pseudo-MOS auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.texture import render_textured_view


LEVELS = ("light", "medium", "heavy")


def load_targets(root: Path) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {}
    for split in ("train", "val", "test", "blind"):
        with np.load(root / f"objective_targets_v2_{split}.npz") as data:
            for name in ("asset_ids", "attacks", "levels", "geometry_quality",
                         "texture_quality", "overall_quality", "objective_noop"):
                chunks.setdefault(name, []).append(data[name].copy())
    return {name: np.concatenate(parts) for name, parts in chunks.items()}


def resolve(data_root: Path, relative: str) -> Path:
    path = data_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def render_card(path: Path, title: str, score: str, size: int) -> Image.Image:
    mesh = GltfReader(path).load_mesh(include_texture=True)
    view = Image.fromarray(render_textured_view(mesh, size=size))
    card = Image.new("RGB", (size, size + 40), "white")
    card.paste(view, (0, 40))
    draw = ImageDraw.Draw(card)
    draw.text((4, 3), title, fill="black")
    draw.text((4, 20), score, fill="black")
    return card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=192)
    args = parser.parse_args()
    values = load_targets(args.target_dir)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    lookup = {(row["asset_id"], row["attack"], row["level"]): row for row in manifest}
    attacks = sorted(set(values["attacks"].astype(str)) - {"clean"})
    selected = []
    rows = []
    for attack in attacks:
        mask = (values["attacks"] == attack) & ~values["objective_noop"].astype(bool)
        medians = {level: float(np.median(values["overall_quality"][mask & (values["levels"] == level)]))
                   for level in LEVELS}
        candidates = sorted(set(values["asset_ids"][mask].astype(str)))
        best = None
        for asset_id in candidates:
            indices = [np.flatnonzero((values["asset_ids"] == asset_id) &
                                      (values["attacks"] == attack) &
                                      (values["levels"] == level) &
                                      ~values["objective_noop"].astype(bool)) for level in LEVELS]
            if not all(len(index) == 1 for index in indices):
                continue
            error = sum(abs(float(values["overall_quality"][index[0]]) - medians[level])
                        for index, level in zip(indices, LEVELS))
            key = (error, asset_id, indices)
            if best is None or key[:2] < best[:2]:
                best = key
        if best is None:
            raise RuntimeError(f"No complete representative sequence: {attack}")
        _, asset_id, indices = best
        clean_row = lookup[(asset_id, "clean", "clean")]
        cards = [render_card(resolve(args.data_root, clean_row["gltf_path"]),
                             f"{attack} / clean", "G 1.000  T 1.000  O 1.000", args.size)]
        case = {"attack": attack, "asset_id": asset_id, "levels": {}}
        for level, index in zip(LEVELS, indices):
            index = int(index[0]); record = lookup[(asset_id, attack, level)]
            quality = {name: float(values[name][index]) for name in
                       ("geometry_quality", "texture_quality", "overall_quality")}
            cards.append(render_card(
                resolve(args.data_root, record["gltf_path"]), level,
                f"G {quality['geometry_quality']:.3f}  T {quality['texture_quality']:.3f}  O {quality['overall_quality']:.3f}",
                args.size,
            ))
            case["levels"][level] = {"gltf_path": record["gltf_path"], **quality}
        row = Image.new("RGB", (4 * args.size, args.size + 40), (230, 230, 230))
        for column, card in enumerate(cards):
            row.paste(card, (column * args.size, 0))
        rows.append(row); selected.append(case)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (4 * args.size, len(rows) * (args.size + 40)), (220, 220, 220))
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * (args.size + 40)))
    sheet.save(args.output_dir / "representative_18_cases.png")
    (args.output_dir / "representative_18_cases.json").write_text(
        json.dumps({"schema_version": 1, "selection": "closest complete asset sequence to per-level attack medians",
                    "cases": selected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"attacks": len(selected), "attacked_cases": len(selected) * 3,
                      "image": str(args.output_dir / "representative_18_cases.png")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
