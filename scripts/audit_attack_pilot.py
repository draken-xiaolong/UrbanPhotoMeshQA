#!/usr/bin/env python3
"""Audit Pilot packages for effective and monotonic degradation; render contact sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.integrity import asset_digest
from urbanphotomeshqa.texture import render_textured_view


LEVEL_ORDER = {"light": 0, "medium": 1, "heavy": 2}


def texture_paths(gltf: Path) -> list[Path]:
    root = json.loads(gltf.read_text(encoding="utf-8"))
    return [(gltf.parent / record["uri"]).resolve() for record in root.get("images", [])]


def texture_distance(source: Image.Image, target: Image.Image) -> float:
    target = target.resize(source.size, Image.Resampling.BILINEAR)
    a = np.asarray(source, dtype=np.float32) / 255.0
    b = np.asarray(target, dtype=np.float32) / 255.0
    return float(np.mean(np.abs(a - b)))


def package_texture_distance(source: Path, target: Path) -> float:
    clean, degraded = texture_paths(source), texture_paths(target)
    if len(clean) != len(degraded) or not clean:
        raise ValueError(f"Texture count mismatch: {source} -> {target}")
    distances = []
    for clean_path, degraded_path in zip(clean, degraded):
        with Image.open(clean_path) as clean_image, Image.open(degraded_path) as degraded_image:
            distances.append(texture_distance(clean_image.convert("RGB"), degraded_image.convert("RGB")))
    return float(np.mean(distances))


def render_card(gltf: Path, label: str, size: int) -> Image.Image:
    asset = GltfReader(gltf).load_mesh(include_texture=True)
    view = Image.fromarray(render_textured_view(asset, size=size))
    card = Image.new("RGB", (size, size + 28), "white")
    card.paste(view, (0, 28))
    ImageDraw.Draw(card).text((5, 7), label, fill="black")
    return card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--preview-assets", nargs="*", default=[])
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    groups = {}
    audit_records = []
    errors = []
    for record in records:
        path = Path(record["gltf_path"])
        digest, dependencies = asset_digest(path)
        if digest != record["asset_digest"]:
            errors.append(f"digest mismatch: {path}")
        source = Path(record["source_gltf"])
        metric = None
        file_effect = None
        if record["attack"].startswith("texture_"):
            file_effect = package_texture_distance(source, path)
            if file_effect <= 0.0:
                errors.append(f"texture no-op: {path}")
            parameters = record["parameters"]
            if record["attack"] == "texture_detail_loss":
                metric = (float(parameters["value"]) if parameters["subtype"] == "gaussian_blur"
                          else 1.0 - float(parameters["value"]))
            elif record["attack"] == "texture_region_missing":
                metric = float(record.get("actual_missing_pixel_fraction_mean",
                                          parameters["missing_pixel_fraction"]))
            else:
                metric = float(parameters["texture_width_shift_fraction"] + parameters["ghost_alpha"])
        elif record["attack"] == "geometry_hole":
            metric = 1.0 - record["face_count"] / record["source_face_count"]
        elif record["attack"] == "mesh_simplification_qem":
            metric = 1.0 - record["face_count"] / record["source_face_count"]
        elif record["attack"] == "geometry_noise_spike":
            source_mesh = GltfReader(source).load_mesh(include_texture=True)
            target_mesh = GltfReader(path).load_mesh(include_texture=True)
            source_to_target = cKDTree(target_mesh.vertices).query(source_mesh.vertices, k=1)[0]
            target_to_source = cKDTree(source_mesh.vertices).query(target_mesh.vertices, k=1)[0]
            metric = max(float(np.max(source_to_target)), float(np.max(target_to_source)))
        key = (record["asset_id"], record["attack"])
        groups.setdefault(key, []).append((LEVEL_ORDER[record["level"]], float(metric)))
        audit_records.append({
            "asset_id": record["asset_id"], "attack": record["attack"], "level": record["level"],
            "effect_metric": float(metric), "file_effect": file_effect,
            "dependencies": len(dependencies), "digest_ok": digest == record["asset_digest"],
        })
    monotonic = {}
    for key, values in groups.items():
        ordered = [metric for _, metric in sorted(values)]
        ok = ordered[0] < ordered[1] and ordered[1] < ordered[2]
        monotonic["/".join(key)] = {"values": ordered, "ok": ok}
        if not ok:
            errors.append(f"non-monotonic: {key} {ordered}")

    args.preview_dir.mkdir(parents=True, exist_ok=True)
    for asset_id in args.preview_assets:
        subset = [record for record in records if record["asset_id"] == asset_id]
        if not subset:
            continue
        source = Path(subset[0]["source_gltf"])
        cards = [render_card(source, "clean", args.size)]
        for record in sorted(subset, key=lambda x: (x["attack"], LEVEL_ORDER[x["level"]])):
            cards.append(render_card(Path(record["gltf_path"]), f"{record['attack']}:{record['level']}", args.size))
        columns = 4
        rows = int(np.ceil(len(cards) / columns))
        sheet = Image.new("RGB", (columns * args.size, rows * (args.size + 28)), (230, 230, 230))
        for index, card in enumerate(cards):
            sheet.paste(card, ((index % columns) * args.size, (index // columns) * (args.size + 28)))
        sheet.save(args.preview_dir / f"{asset_id}_contact_sheet.png")

    report = {
        "schema_version": 1, "variants": len(records), "errors": errors,
        "passed": not errors, "monotonic": monotonic, "records": audit_records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "variants": len(records), "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
