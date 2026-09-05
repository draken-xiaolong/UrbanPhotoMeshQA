#!/usr/bin/env python3
"""Build one auditable Iteration-3 building pilot with eight atomic and one joint degradation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.gltf_export import export_textured_gltf
from urbanphotomeshqa.integrity import asset_digest
from urbanphotomeshqa.mesh_attacks import connected_face_patch, recompute_vertex_normals
from urbanphotomeshqa.real_attacks import (
    geometry_hole,
    geometry_noise_spike,
    qem_simplify_textured,
    texture_detail_loss,
    texture_misalignment,
    texture_region_missing,
)
from urbanphotomeshqa.texture import render_textured_view


SEED = 2026

LEVEL_BY_VARIANT = {
    "geometry_hole": "medium",
    "mesh_simplification_qem": "light",
    "geometry_noise_spike": "heavy",
    "geometry_over_smoothing": "medium",
    "texture_detail_loss": "light",
    "texture_region_missing": "heavy",
    "texture_misalignment": "medium",
    "texture_seam_radiometric": "light",
    "geometry_texture_combined_sample": "heavy",
}

INITIAL_GRADE_BY_VARIANT = {
    "geometry_hole": 2,
    "mesh_simplification_qem": 4,
    "geometry_noise_spike": 4,
    "geometry_over_smoothing": 4,
    "texture_detail_loss": 5,
    "texture_region_missing": 1,
    "texture_misalignment": 4,
    "texture_seam_radiometric": 5,
    "geometry_texture_combined_sample": 3,
}


def stable_seed(*values: object) -> int:
    payload = "|".join(map(str, (SEED, *values))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def stage_clean(source: Path, destination: Path) -> Path:
    root = json.loads(source.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(root.get("buffers", [])):
        uri = record.get("uri")
        if not uri or uri.startswith("data:"):
            raise ValueError(f"Unsupported buffer URI: {uri}")
        name = f"buffer_{index:02d}{Path(uri).suffix or '.bin'}"
        shutil.copy2((source.parent / uri).resolve(), destination / name)
        record["uri"] = name
    texture_dir = destination / "textures"
    texture_dir.mkdir(exist_ok=True)
    for index, record in enumerate(root.get("images", [])):
        uri = record.get("uri")
        if not uri or uri.startswith("data:"):
            raise ValueError(f"Unsupported image URI: {uri}")
        suffix = Path(uri).suffix.lower() or ".png"
        name = f"image_{index:02d}{suffix}"
        shutil.copy2((source.parent / uri).resolve(), texture_dir / name)
        record["uri"] = f"textures/{name}"
    output = destination / "clean_scale5.gltf"
    output.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def prepare_textures(asset, destination: Path) -> list[str]:
    texture_dir = destination / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    names = []
    paths = asset.metadata.get("material_texture_paths", [])
    material_count = max(1, int(asset.metadata.get("material_count", len(paths))))
    for index in range(material_count):
        source = Path(paths[index]) if index < len(paths) and paths[index] else None
        if source is not None and source.is_file():
            name = f"material_{index:02d}{source.suffix.lower()}"
            shutil.copy2(source, texture_dir / name)
        else:
            name = f"material_{index:02d}.png"
            Image.new("RGBA", (8, 8), (180, 180, 180, 255)).save(texture_dir / name)
        names.append(name)
    return names


def export_geometry(asset, attacked, destination: Path, filename: str = "model.gltf") -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    origin = 0.5 * (asset.vertices.min(axis=0) + asset.vertices.max(axis=0))
    names = prepare_textures(asset, destination)
    output = destination / filename
    export_textured_gltf(attacked, output, names, coordinate_origin=origin)
    return output


def over_smooth(asset, affected_fraction: float, iterations: int, strength: float, seed: int):
    faces = np.asarray(asset.faces, dtype=np.int64)
    selected_faces = connected_face_patch(
        faces, max(1, int(round(len(faces) * affected_fraction))), np.random.default_rng(seed)
    )
    selected = np.unique(faces[selected_faces])
    selected_mask = np.zeros(len(asset.vertices), dtype=bool)
    selected_mask[selected] = True
    neighbors = [set() for _ in range(len(asset.vertices))]
    for a, b, c in faces:
        neighbors[a].update((int(b), int(c)))
        neighbors[b].update((int(a), int(c)))
        neighbors[c].update((int(a), int(b)))
    vertices = np.asarray(asset.vertices, dtype=np.float64).copy()
    for _ in range(iterations):
        updated = vertices.copy()
        for index in selected:
            adjacent = list(neighbors[int(index)])
            if adjacent:
                mean = vertices[adjacent].mean(axis=0)
                updated[index] = (1.0 - strength) * vertices[index] + strength * mean
        vertices = updated
    return replace(
        asset,
        vertices=vertices,
        normals=recompute_vertex_normals(vertices, faces),
        metadata={**asset.metadata, "attack": "geometry_over_smoothing"},
    ), selected_mask


def rewrite_textures(source: Path, destination: Path, transform, filename: str = "model.gltf") -> Path:
    root = json.loads(source.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(root.get("buffers", [])):
        uri = record["uri"]
        name = f"buffer_{index:02d}{Path(uri).suffix or '.bin'}"
        shutil.copy2((source.parent / uri).resolve(), destination / name)
        record["uri"] = name
    texture_dir = destination / "textures"
    texture_dir.mkdir(exist_ok=True)
    for index, record in enumerate(root.get("images", [])):
        image = Image.open((source.parent / record["uri"]).resolve()).convert("RGBA")
        result = transform(image, index)
        name = f"image_{index:02d}.png"
        result.save(texture_dir / name, format="PNG", optimize=False)
        record["uri"] = f"textures/{name}"
    output = destination / filename
    output.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def radiometric_seam(image: Image.Image, _: int) -> Image.Image:
    array = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    height, width = array.shape[:2]
    left, right = int(width * 0.34), int(width * 0.70)
    top, bottom = int(height * 0.18), int(height * 0.82)
    region = Image.fromarray(array[top:bottom, left:right])
    region = ImageEnhance.Brightness(region).enhance(1.12)
    changed = np.asarray(region, dtype=np.uint8).copy()
    changed[..., 0] = np.clip(changed[..., 0].astype(np.int16) + 8, 0, 255)
    changed[..., 2] = np.clip(changed[..., 2].astype(np.int16) - 5, 0, 255)
    array[top:bottom, left:right] = changed
    return Image.fromarray(array)


def validate(path: Path) -> dict:
    digest, dependencies = asset_digest(path)
    mesh = GltfReader(path).load_mesh(include_texture=True)
    if not np.isfinite(mesh.vertices).all() or len(mesh.faces) < 1:
        raise ValueError(f"Invalid generated mesh: {path}")
    return {
        "asset_digest": digest,
        "dependency_count": len(dependencies),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "image_count": int(mesh.metadata["image_count"]),
    }


def render_card(path: Path, label: str, size: int = 320) -> Image.Image:
    mesh = GltfReader(path).load_mesh(include_texture=True)
    view = Image.fromarray(render_textured_view(mesh, size=size))
    card = Image.new("RGB", (size, size + 34), "white")
    card.paste(view, (0, 34))
    ImageDraw.Draw(card).text((6, 8), label, fill="black")
    return card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    args = parser.parse_args()

    asset_root = args.output_root / "assets" / args.asset_id
    manifest_root = args.output_root / "manifests"
    audit_root = args.output_root / "audits"
    preview_root = args.output_root / "previews"
    for path in (manifest_root, audit_root, preview_root, args.output_root / "human_ratings",
                 args.output_root / "patch_targets"):
        path.mkdir(parents=True, exist_ok=True)

    clean_path = stage_clean(args.source, asset_root / "clean")
    clean = GltfReader(clean_path).load_mesh(include_texture=True)
    variants: list[tuple[str, Path, dict]] = [("clean", clean_path, {"quality_grade": 5})]

    geometry_specs = {
        "geometry_hole": (geometry_hole(clean, 0.15, stable_seed(args.asset_id, "hole")), {"removed_face_fraction": 0.15}),
        "mesh_simplification_qem": (qem_simplify_textured(clean, 0.85), {"retained_face_fraction": 0.85}),
        "geometry_noise_spike": (geometry_noise_spike(clean, 0.006, 0.30, stable_seed(args.asset_id, "noise")), {"bbox_diagonal_fraction": 0.006, "affected_face_fraction": 0.30}),
    }
    smoothed, selected_mask = over_smooth(clean, 0.35, 8, 0.35, stable_seed(args.asset_id, "smooth"))
    geometry_specs["geometry_over_smoothing"] = (smoothed, {"affected_vertex_count": int(selected_mask.sum()), "iterations": 8, "strength": 0.35})
    for name, (mesh, parameters) in geometry_specs.items():
        scale = f"scale{INITIAL_GRADE_BY_VARIANT[name]}"
        path = export_geometry(
            clean, mesh, asset_root / "degradations" / scale / name,
            f"{name}_{scale}.gltf",
        )
        variants.append((name, path, parameters))

    texture_specs = {
        "texture_detail_loss": lambda image, _: texture_detail_loss(image, "gaussian_blur", 0.8),
        "texture_region_missing": lambda image, index: texture_region_missing(image, 0.30, stable_seed(args.asset_id, "missing", index))[0],
        "texture_misalignment": lambda image, index: texture_misalignment(image, 0.025, 0.45, stable_seed(args.asset_id, "misalignment", index)),
        "texture_seam_radiometric": radiometric_seam,
    }
    texture_parameters = {
        "texture_detail_loss": {"subtype": "gaussian_blur", "radius": 0.8},
        "texture_region_missing": {"missing_pixel_fraction": 0.30},
        "texture_misalignment": {"shift_fraction": 0.025, "ghost_alpha": 0.45},
        "texture_seam_radiometric": {"brightness": 1.12, "red_offset": 8, "blue_offset": -5},
    }
    for name, transform in texture_specs.items():
        initial_grade = INITIAL_GRADE_BY_VARIANT[name]
        scale = f"scale{initial_grade}"
        path = rewrite_textures(
            clean_path, asset_root / "degradations" / scale / name, transform,
            f"{name}_{scale}.gltf",
        )
        variants.append((name, path, texture_parameters[name]))

    combined_stage = asset_root / "_combined_geometry_stage"
    qem_path = export_geometry(clean, qem_simplify_textured(clean, 0.55), combined_stage)
    combined_path = rewrite_textures(
        qem_path,
        asset_root / "degradations" / "scale3" / "geometry_texture_combined_sample",
        lambda image, _: texture_detail_loss(image, "gaussian_blur", 4.0),
        "geometry_texture_combined_sample_scale3.gltf",
    )
    variants.append(("geometry_texture_combined_sample", combined_path, {
        "geometry": "mesh_simplification_qem", "retained_face_fraction": 0.55,
        "texture": "texture_detail_loss", "gaussian_blur_radius": 4.0,
    }))
    shutil.rmtree(combined_stage, ignore_errors=True)

    records = []
    cards = []
    for name, path, parameters in variants:
        status = validate(path)
        metadata = {
            "schema_version": 1,
            "asset_id": args.asset_id,
            "variant": name,
            "generation_level": "clean" if name == "clean" else LEVEL_BY_VARIANT[name],
            "human_rating_initial": 5 if name == "clean" else INITIAL_GRADE_BY_VARIANT[name],
            "quality_scale": "scale5" if name == "clean" else f"scale{INITIAL_GRADE_BY_VARIANT[name]}",
            "parameters": parameters,
            "gltf_path": str(path),
            **status,
        }
        (path.parent / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(metadata)
        cards.append(render_card(path, f"{name} [{metadata['generation_level']}]"))

    manifest = {"schema_version": 1, "seed": SEED, "pilot": True, "records": records}
    manifest_path = manifest_root / f"{args.asset_id}_pilot.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "status": "PASSED", "asset_id": args.asset_id, "versions": len(records),
        "all_parseable": True, "manifest": str(manifest_path),
    }
    (audit_root / f"{args.asset_id}_pilot_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = 2
    rows = int(np.ceil(len(cards) / columns))
    sheet = Image.new("RGB", (columns * 320, rows * 354), (232, 232, 232))
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % columns) * 320, (index // columns) * 354))
    preview = preview_root / f"{args.asset_id}_pilot_contact_sheet.png"
    sheet.save(preview)
    print(json.dumps({**audit, "preview": str(preview)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
