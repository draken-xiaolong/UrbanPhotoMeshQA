#!/usr/bin/env python3
"""Export reloadable glTF packages for the six first-stage quality degradations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from dataclasses import replace

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.gltf_export import export_textured_gltf
from urbanphotomeshqa.integrity import asset_digest, extractor_signature, sha256_file
from urbanphotomeshqa.real_attacks import (
    geometry_hole,
    geometry_noise_spike,
    qem_simplify_textured,
    texture_detail_loss,
    texture_misalignment,
    texture_region_missing,
)


LEVELS = ("light", "medium", "heavy")


def stable_seed(seed: int, *values: object) -> int:
    payload = "|".join([str(seed), *(str(value) for value in values)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def texture_subtype(asset_id: str) -> str:
    return ("gaussian_blur", "texture_downsample")[hashlib.sha256(asset_id.encode()).digest()[0] & 1]


def locate_source(source_root: Path, record: dict) -> Path:
    return source_root / record["sheet"] / record.get("class_name", "BUILDING") / record["asset_id"] / f"{record['asset_id']}.gltf"


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    seen = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload["records"]:
            key = record["asset_id"]
            if key not in seen:
                records.append(record)
                seen.add(key)
    return records


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


def export_geometry(source: Path, output: Path, attack: str, params: dict, seed: int) -> dict:
    source_asset = GltfReader(source).load_mesh(include_texture=True)
    coordinate_origin = 0.5 * (
        source_asset.vertices.min(axis=0) + source_asset.vertices.max(axis=0)
    )
    if source_asset.texcoords is None:
        source_asset = replace(source_asset, texcoords=np.zeros((len(source_asset.vertices), 2), np.float64))
    elif not np.isfinite(source_asset.texcoords).all():
        source_asset = replace(source_asset, texcoords=np.nan_to_num(source_asset.texcoords, nan=0.0))
    if attack == "geometry_hole":
        attacked = geometry_hole(source_asset, params["removed_face_fraction"], seed)
    elif attack == "mesh_simplification_qem":
        attacked = qem_simplify_textured(source_asset, params["retained_face_fraction"])
    elif attack == "geometry_noise_spike":
        attacked = geometry_noise_spike(
            source_asset, params["bbox_diagonal_fraction"], params["affected_face_fraction"], seed
        )
    else:
        raise ValueError(attack)
    names = prepare_textures(source_asset, output.parent)
    export_textured_gltf(attacked, output, names, coordinate_origin=coordinate_origin)
    return {
        "source_vertex_count": len(source_asset.vertices),
        "source_face_count": len(source_asset.faces),
        "generated_vertex_count": len(attacked.vertices),
        "generated_face_count": len(attacked.faces),
    }


def export_texture(source: Path, output: Path, attack: str, params: dict, seed: int) -> dict:
    root = json.loads(source.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    for buffer in root.get("buffers", []):
        uri = buffer.get("uri")
        if not uri or uri.startswith("data:"):
            raise ValueError(f"Unsupported embedded buffer: {source}")
        target_name = Path(uri).name
        shutil.copy2((source.parent / uri).resolve(), output.parent / target_name)
        buffer["uri"] = target_name
    actual_values = []
    for index, image_record in enumerate(root.get("images", [])):
        uri = image_record.get("uri")
        if not uri or uri.startswith("data:"):
            raise ValueError(f"Unsupported embedded image: {source}")
        image = Image.open((source.parent / uri).resolve())
        image_seed = stable_seed(seed, index)
        if attack == "texture_detail_loss":
            degraded = texture_detail_loss(image, params["subtype"], params["value"])
        elif attack == "texture_region_missing":
            degraded, actual = texture_region_missing(image, params["missing_pixel_fraction"], image_seed)
            actual_values.append(actual)
        elif attack == "texture_misalignment":
            degraded = texture_misalignment(
                image, params["texture_width_shift_fraction"], params["ghost_alpha"], image_seed
            )
        else:
            raise ValueError(attack)
        target_name = f"image_{index:02d}.png"
        target = output.parent / "textures" / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        degraded.save(target, format="PNG", optimize=False)
        image_record["uri"] = f"textures/{target_name}"
    output.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    if actual_values:
        return {"actual_missing_pixel_fraction_mean": float(np.mean(actual_values))}
    return {}


def validate_output(path: Path, source: Path, attack: str) -> dict:
    generated_digest, dependencies = asset_digest(path)
    mesh = GltfReader(path).load_mesh(include_texture=True)
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Non-finite vertices: {path}")
    if len(mesh.faces) < 1 or np.max(mesh.faces) >= len(mesh.vertices):
        raise ValueError(f"Invalid faces: {path}")
    source_mesh = GltfReader(source).load_mesh(include_texture=True)
    if attack.startswith("texture_"):
        if not np.array_equal(mesh.faces, source_mesh.faces) or not np.allclose(mesh.vertices, source_mesh.vertices):
            raise ValueError(f"Texture attack modified geometry: {path}")
        if not np.array_equal(np.isfinite(mesh.texcoords), np.isfinite(source_mesh.texcoords)):
            raise ValueError(f"Texture attack modified UV availability: {path}")
    elif not np.isfinite(mesh.texcoords).all():
        raise ValueError(f"Geometry export contains non-finite UV: {path}")
    return {
        "asset_digest": generated_digest,
        "dependencies": dependencies,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "material_count": int(mesh.metadata["material_count"]),
        "image_count": int(mesh.metadata["image_count"]),
    }


def parameters(config: dict, attack: str, level_index: int, asset_id: str) -> dict:
    spec = config["attacks"][attack]
    if attack == "geometry_hole":
        return {"removed_face_fraction": spec["removed_face_fraction"][level_index]}
    if attack == "mesh_simplification_qem":
        return {"retained_face_fraction": spec["retained_face_fraction"][level_index]}
    if attack == "geometry_noise_spike":
        return {key: spec[key][level_index] for key in ("bbox_diagonal_fraction", "affected_face_fraction")}
    if attack == "texture_detail_loss":
        subtype = texture_subtype(asset_id)
        key = "gaussian_blur_radius" if subtype == "gaussian_blur" else "texture_resolution_fraction"
        return {"subtype": subtype, "value": spec[key][level_index]}
    if attack == "texture_region_missing":
        return {"missing_pixel_fraction": spec["missing_pixel_fraction"][level_index]}
    if attack == "texture_misalignment":
        return {key: spec[key][level_index] for key in ("texture_width_shift_fraction", "ghost_alpha")}
    raise ValueError(attack)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--asset-ids", nargs="*")
    parser.add_argument("--attacks", nargs="*", choices=(
        "geometry_hole", "mesh_simplification_qem", "geometry_noise_spike",
        "texture_detail_loss", "texture_region_missing", "texture_misalignment",
    ))
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    project = Path(__file__).resolve().parents[1]
    generator_signature = extractor_signature({
        "schema_version": 2,
        "files": {
            str(path.relative_to(project)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                project / "src/urbanphotomeshqa/gltf.py",
                project / "src/urbanphotomeshqa/gltf_export.py",
                project / "src/urbanphotomeshqa/real_attacks.py",
            )
        },
    })
    records = load_records(args.manifests)
    if args.asset_ids:
        requested = set(args.asset_ids)
        records = [record for record in records if record["asset_id"] in requested]
        missing = requested - {record["asset_id"] for record in records}
        if missing:
            raise SystemExit(f"Unknown asset ids: {sorted(missing)}")
    output_records = []
    attacks = tuple(args.attacks or config["attacks"])
    for record in records:
        source = locate_source(args.source_root, record)
        source_digest, _ = asset_digest(source)
        for attack in attacks:
            for level_index, level in enumerate(LEVELS):
                params = parameters(config, attack, level_index, record["asset_id"])
                # Keep the damaged location/realisation fixed across severity
                # levels; only the magnitude changes. This prevents severity
                # supervision from being confounded by random spatial changes.
                seed = stable_seed(config["seed"], record["asset_id"], attack)
                output_dir = args.output_root / record["sheet"] / record.get("class_name", "BUILDING") / record["asset_id"] / attack / level
                output = output_dir / f"{record['asset_id']}.gltf"
                metadata_file = output_dir / "metadata.json"
                if output.is_file() and metadata_file.is_file():
                    try:
                        previous = json.loads(metadata_file.read_text(encoding="utf-8"))
                        if (previous.get("source_asset_digest") == source_digest
                                and previous.get("generator_signature") == generator_signature
                                and previous.get("parameters") == params
                                and previous.get("seed") == seed
                                and previous.get("attack") == attack
                                and previous.get("level") == level):
                            current_digest, _ = asset_digest(output)
                            if current_digest == previous.get("asset_digest"):
                                output_records.append({**previous, "gltf_path": str(output)})
                                print(f"reuse {record['asset_id']} {attack} {level}", flush=True)
                                continue
                    except Exception:
                        pass
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                if attack.startswith("texture_"):
                    generation = export_texture(source, output, attack, params, seed)
                else:
                    generation = export_geometry(source, output, attack, params, seed)
                validation = validate_output(output, source, attack)
                metadata = {
                    "schema_version": 2, "generator_signature": generator_signature,
                    "asset_id": record["asset_id"], "sheet": record["sheet"],
                    "split": record["split"], "attack": attack, "level": level, "parameters": params,
                    "seed": seed, "source_gltf": str(source), "source_asset_digest": source_digest,
                    **generation, **validation,
                }
                metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                output_records.append({**metadata, "gltf_path": str(output)})
                print(f"ok {record['asset_id']} {attack} {level}", flush=True)
    payload = {"schema_version": 2, "seed": config["seed"],
               "generator_signature": generator_signature, "records": output_records}
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"assets": len(records), "variants": len(output_records), "manifest": str(args.manifest_output)}))


if __name__ == "__main__":
    main()
