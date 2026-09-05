#!/usr/bin/env python3
"""Generate the frozen-candidate 50-version V3 pilot for one building."""

from __future__ import annotations

import argparse
import io
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from build_iteration3_pilot import (
    export_geometry, over_smooth, rewrite_textures, stable_seed, validate,
)
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.mesh_attacks import recompute_vertex_normals
from urbanphotomeshqa.real_attacks import (
    geometry_hole, geometry_noise_spike, qem_simplify_textured,
    texture_misalignment, texture_region_missing,
)

SEED = 2026
LEVEL_TO_SCALE = {1: 4, 2: 3, 3: 2, 4: 1}


def quantize_positions(asset, bins: int):
    vertices = np.asarray(asset.vertices, dtype=np.float64)
    lower = vertices.min(axis=0)
    span = np.maximum(vertices.max(axis=0) - lower, 1e-12)
    quantized = lower + np.rint((vertices - lower) / span * bins) / bins * span
    return replace(asset, vertices=quantized,
                   normals=recompute_vertex_normals(quantized, asset.faces))


def quantize_uv(asset, bins: int):
    uv = np.asarray(asset.texcoords, dtype=np.float64)
    return replace(asset, texcoords=np.rint(uv * bins) / bins)


def blur_resolution(factor: float, radius: float):
    def apply(image: Image.Image, _: int) -> Image.Image:
        image = image.convert("RGBA")
        small = image.resize((max(1, round(image.width * factor)),
                              max(1, round(image.height * factor))), Image.Resampling.LANCZOS)
        restored = small.resize(image.size, Image.Resampling.BILINEAR)
        return restored.filter(ImageFilter.GaussianBlur(radius))
    return apply


def jpeg_compression(quality: int):
    def apply(image: Image.Image, _: int) -> Image.Image:
        alpha = image.convert("RGBA").getchannel("A")
        stream = io.BytesIO()
        image.convert("RGB").save(stream, "JPEG", quality=quality, subsampling=2)
        stream.seek(0)
        decoded = Image.open(stream).convert("RGBA")
        decoded.putalpha(alpha)
        return decoded
    return apply


def missing(fraction: float, key: str):
    return lambda image, index: texture_region_missing(
        image, fraction, stable_seed(key, index))[0]


def misalignment(shift: float, alpha: float, key: str):
    return lambda image, index: texture_misalignment(
        image, shift, alpha, stable_seed(key, index))


def seam(brightness: float, offset: int):
    def apply(image: Image.Image, _: int) -> Image.Image:
        array = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
        h, w = array.shape[:2]
        top, bottom, left, right = int(.15*h), int(.85*h), int(.32*w), int(.70*w)
        region = Image.fromarray(array[top:bottom, left:right])
        changed = np.asarray(ImageEnhance.Brightness(region).enhance(brightness), dtype=np.uint8).copy()
        changed[..., 0] = np.clip(changed[..., 0].astype(np.int16) + offset, 0, 255)
        changed[..., 2] = np.clip(changed[..., 2].astype(np.int16) - offset//2, 0, 255)
        array[top:bottom, left:right] = changed
        return Image.fromarray(array)
    return apply


def compose(*transforms):
    def apply(image: Image.Image, index: int) -> Image.Image:
        for transform in transforms:
            image = transform(image, index)
        return image
    return apply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    args = parser.parse_args()

    output = args.asset_root / "versions_50"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing pilot: {output}")
    output.mkdir(parents=True)
    stage = args.asset_root / "_versions_50_stage"
    stage.mkdir(parents=True, exist_ok=True)
    clean = GltfReader(args.clean).load_mesh(include_texture=True)
    records = []

    def emit(name: str, scale: int, parameters: dict, mesh=None, transform=None):
        destination = output / name
        if mesh is not None:
            base = export_geometry(clean, mesh, stage / name)
            path = (rewrite_textures(base, destination, transform, f"scale{scale}_{name}.gltf")
                    if transform else export_geometry(clean, mesh, destination, f"scale{scale}_{name}.gltf"))
        else:
            path = rewrite_textures(args.clean, destination, transform, f"scale{scale}_{name}.gltf")
        status = validate(path)
        metadata = {"schema_version": 2, "seed": SEED, "asset_id": args.asset_id,
                    "variant": name, "human_rating_initial": scale,
                    "parameters": parameters, "gltf_path": str(path), **status}
        (destination / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(metadata)

    # Clean is referenced, not duplicated in canonical storage.
    records.append({"schema_version": 2, "seed": SEED, "asset_id": args.asset_id,
                    "variant": "clean", "human_rating_initial": 5,
                    "parameters": {}, "gltf_path": str(args.clean), **validate(args.clean)})

    holes = [.005, .02, .06, .15]
    noises = [(.0005, .08), (.0015, .15), (.004, .25), (.008, .40)]
    retains = [.85, .60, .35, .15]
    smooths = [(.12, 3, .18), (.22, 6, .28), (.35, 10, .38), (.50, 16, .50)]
    bins = [65535, 8191, 2047, 511]
    blur_levels = [(.85, .35), (.60, .8), (.35, 1.6), (.18, 3.2)]
    jpeg_levels = [85, 60, 35, 12]
    missing_levels = [.015, .05, .15, .30]
    align_levels = [(.003, .20), (.008, .30), (.018, .45), (.035, .60)]
    seam_levels = [(1.04, 3), (1.10, 7), (1.20, 14), (1.35, 24)]

    for level in range(1, 5):
        scale = LEVEL_TO_SCALE[level]
        suffix = f"level{level}"
        emit(f"geometry_missing_{suffix}", scale, {"removed_fraction": holes[level-1]},
             geometry_hole(clean, holes[level-1], stable_seed(args.asset_id, "hole", level)))
        diag, affected = noises[level-1]
        emit(f"geometry_artifacts_{suffix}", scale,
             {"diagonal_fraction": diag, "affected_face_fraction": affected},
             geometry_noise_spike(clean, diag, affected, stable_seed(args.asset_id, "noise", level)))
        emit(f"mesh_simplification_qem_{suffix}", scale,
             {"retained_fraction": retains[level-1]}, qem_simplify_textured(clean, retains[level-1]))
        fraction, iterations, strength = smooths[level-1]
        smoothed, _ = over_smooth(clean, fraction, iterations, strength,
                                  stable_seed(args.asset_id, "smooth", level))
        emit(f"geometry_smoothing_{suffix}", scale,
             {"affected_fraction": fraction, "iterations": iterations, "strength": strength}, smoothed)
        emit(f"position_quantization_{suffix}", scale, {"bins": bins[level-1]},
             quantize_positions(clean, bins[level-1]))
        factor, radius = blur_levels[level-1]
        emit(f"texture_blur_resolution_loss_{suffix}", scale,
             {"downsample_factor": factor, "blur_radius": radius},
             transform=blur_resolution(factor, radius))
        emit(f"texture_compression_{suffix}", scale, {"jpeg_quality": jpeg_levels[level-1]},
             transform=jpeg_compression(jpeg_levels[level-1]))
        emit(f"texture_missing_occlusion_{suffix}", scale,
             {"missing_fraction": missing_levels[level-1]},
             transform=missing(missing_levels[level-1], f"missing-{level}"))
        shift, alpha = align_levels[level-1]
        emit(f"texture_misalignment_uv_{suffix}", scale,
             {"shift_fraction": shift, "ghost_alpha": alpha},
             transform=misalignment(shift, alpha, f"align-{level}"))
        brightness, offset = seam_levels[level-1]
        emit(f"texture_seam_radiometric_{suffix}", scale,
             {"brightness": brightness, "red_offset": offset}, transform=seam(brightness, offset))

    # Nine realistic representative combinations; initial scales are only review suggestions.
    qem60 = qem_simplify_textured(clean, .60)
    qem45 = qem_simplify_textured(clean, .45)
    pos = quantize_positions(clean, 2047)
    uv = quantize_uv(qem45, 1023)
    hole = geometry_hole(clean, .06, stable_seed(args.asset_id, "combo-hole"))
    artifact = geometry_noise_spike(clean, .004, .25, stable_seed(args.asset_id, "combo-noise"))
    smooth, _ = over_smooth(clean, .35, 10, .38, stable_seed(args.asset_id, "combo-smooth"))
    combos = [
        ("combined_qem_texture_downsampling", qem60, blur_resolution(.45, .6), 3, {"qem_retained": .60, "downsample": .45}),
        ("combined_position_quantization_texture_compression", pos, jpeg_compression(35), 3, {"position_bins": 2047, "jpeg_quality": 35}),
        ("combined_qem_uv_texture_compression", uv, jpeg_compression(35), 2, {"qem_retained": .45, "uv_bins": 1023, "jpeg_quality": 35}),
        ("combined_geometry_missing_texture_missing", hole, missing(.15, "combo-missing"), 2, {"hole_fraction": .06, "texture_missing": .15}),
        ("combined_geometry_artifacts_texture_misalignment", artifact, misalignment(.018, .45, "combo-align"), 2, {"noise": .004, "shift": .018}),
        ("combined_geometry_smoothing_texture_blur", smooth, blur_resolution(.35, 1.6), 2, {"smoothing": .38, "downsample": .35}),
        ("combined_geometry_missing_texture_misalignment", hole, misalignment(.018, .45, "combo-hole-align"), 2, {"hole_fraction": .06, "shift": .018}),
        ("combined_texture_blur_radiometric_seam", None, compose(blur_resolution(.45, 1.2), seam(1.20, 14)), 3, {"downsample": .45, "brightness": 1.20}),
        ("combined_qem_texture_compression", qem60, jpeg_compression(35), 3, {"qem_retained": .60, "jpeg_quality": 35}),
    ]
    for name, mesh, transform, scale, parameters in combos:
        emit(name, scale, parameters, mesh=mesh, transform=transform)

    shutil.rmtree(stage, ignore_errors=True)
    manifest = {"schema_version": 2, "seed": SEED, "protocol": "v3_50_versions_draft",
                "asset_id": args.asset_id, "records": records}
    manifest_path = args.asset_root / "versions_50_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASSED", "versions": len(records),
                      "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
