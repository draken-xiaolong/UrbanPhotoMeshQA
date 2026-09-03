#!/usr/bin/env python3
"""Audit geometry-attack exports for coordinate precision and level validity."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.integrity import asset_digest, sha256_file
from urbanphotomeshqa.real_attacks import (
    geometry_hole,
    geometry_noise_spike,
    qem_simplify_textured,
)


LEVEL_ORDER = {"light": 0, "medium": 1, "heavy": 2}


def expected_attack(source, record):
    parameters = record["parameters"]
    if record["attack"] == "geometry_hole":
        return geometry_hole(source, parameters["removed_face_fraction"], record["seed"])
    if record["attack"] == "mesh_simplification_qem":
        return qem_simplify_textured(source, parameters["retained_face_fraction"])
    if record["attack"] == "geometry_noise_spike":
        return geometry_noise_spike(
            source,
            parameters["bbox_diagonal_fraction"],
            parameters["affected_face_fraction"],
            record["seed"],
        )
    raise ValueError(record["attack"])


def symmetric_vertex_error(expected: np.ndarray, observed: np.ndarray) -> float:
    forward = cKDTree(expected).query(observed, k=1)[0]
    reverse = cKDTree(observed).query(expected, k=1)[0]
    return max(float(forward.max(initial=0.0)), float(reverse.max(initial=0.0)))


def material_semantics(profile: dict) -> dict:
    pbr = profile.get("pbrMetallicRoughness", {})
    return {
        "doubleSided": bool(profile.get("doubleSided", False)),
        "alphaMode": profile.get("alphaMode", "OPAQUE"),
        "alphaCutoff": profile.get("alphaCutoff"),
        "emissiveFactor": profile.get("emissiveFactor"),
        "baseColorFactor": pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]),
        "metallicFactor": pbr.get("metallicFactor", 1.0),
        "roughnessFactor": pbr.get("roughnessFactor", 1.0),
        "baseColorSampler": profile.get("baseColorSampler"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    source_cache = {}
    audit_rows = []
    errors = []
    for index, record in enumerate(records):
        asset_id = record["asset_id"]
        if asset_id not in source_cache:
            source = GltfReader(record["source_gltf"]).load_mesh(include_texture=True)
            if source.texcoords is None:
                source = replace(source, texcoords=np.zeros((len(source.vertices), 2), np.float64))
            elif not np.isfinite(source.texcoords).all():
                source = replace(source, texcoords=np.nan_to_num(source.texcoords, nan=0.0))
            source_cache[asset_id] = source
        source = source_cache[asset_id]
        expected = expected_attack(source, record)
        output_path = Path(record["gltf_path"])
        observed_reader = GltfReader(output_path)
        observed = observed_reader.load_mesh(include_texture=True)
        digest, dependencies = asset_digest(output_path)

        coordinate_error = symmetric_vertex_error(expected.vertices, observed.vertices)
        position_accessors = []
        for mesh in observed_reader.root.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                accessor = primitive.get("attributes", {}).get("POSITION")
                if accessor is not None:
                    position_accessors.append(observed_reader.accessor(int(accessor)))
        local_position_max_abs = max(
            (float(np.abs(value).max(initial=0.0)) for value in position_accessors),
            default=0.0,
        )
        expected_materials = [material_semantics(value)
                              for value in source.metadata.get("material_profiles", [])]
        observed_materials = [material_semantics(value)
                              for value in observed.metadata.get("material_profiles", [])]
        used_materials = sorted(set(int(value) for value in expected.face_materials))
        materials_preserved = all(
            material < len(expected_materials)
            and material < len(observed_materials)
            and expected_materials[material] == observed_materials[material]
            for material in used_materials
        )
        texture_hashes_preserved = True
        for material in used_materials:
            source_path = source.metadata["material_texture_paths"][material]
            output_path_texture = observed.metadata["material_texture_paths"][material]
            if source_path and output_path_texture:
                texture_hashes_preserved &= sha256_file(Path(source_path)) == sha256_file(
                    Path(output_path_texture)
                )

        row_errors = []
        if digest != record["asset_digest"]:
            row_errors.append("asset_digest_changed")
        if coordinate_error > args.coordinate_tolerance:
            row_errors.append("coordinate_roundtrip_error")
        if not materials_preserved:
            row_errors.append("material_semantics_changed")
        if not texture_hashes_preserved:
            row_errors.append("texture_bytes_changed")
        if len(dependencies) < 3:
            row_errors.append("missing_dependency")
        errors.extend(f"{asset_id}/{record['attack']}/{record['level']}: {value}"
                      for value in row_errors)
        audit_rows.append({
            "asset_id": asset_id,
            "attack": record["attack"],
            "level": record["level"],
            "coordinate_roundtrip_max_m": coordinate_error,
            "local_position_max_abs_m": local_position_max_abs,
            "face_count": int(len(observed.faces)),
            "asset_digest": digest,
            "materials_preserved": materials_preserved,
            "texture_hashes_preserved": bool(texture_hashes_preserved),
            "errors": row_errors,
        })
        print(f"audit {index + 1}/{len(records)} {asset_id} "
              f"{record['attack']} {record['level']} error={coordinate_error:.3g}", flush=True)

    groups = {}
    for row in audit_rows:
        groups.setdefault((row["asset_id"], row["attack"]), []).append(row)
    level_checks = {}
    for (asset_id, attack), rows in groups.items():
        rows.sort(key=lambda row: LEVEL_ORDER[row["level"]])
        digests_unique = len({row["asset_digest"] for row in rows}) == len(rows)
        face_counts = [row["face_count"] for row in rows]
        if attack in {"geometry_hole", "mesh_simplification_qem"}:
            monotonic = face_counts[0] > face_counts[1] > face_counts[2]
        else:
            source = source_cache[asset_id]
            displacements = []
            source_records = sorted(
                [record for record in records
                 if record["asset_id"] == asset_id and record["attack"] == attack],
                key=lambda record: LEVEL_ORDER[record["level"]],
            )
            for record in source_records:
                expected = expected_attack(source, record)
                displacements.append(float(np.linalg.norm(
                    expected.vertices - source.vertices, axis=1).max(initial=0.0)))
            monotonic = displacements[0] < displacements[1] < displacements[2]
        key = f"{asset_id}/{attack}"
        level_checks[key] = {
            "digests_unique": digests_unique,
            "severity_monotonic": monotonic,
            "face_counts": face_counts,
        }
        if not digests_unique:
            errors.append(f"{key}: duplicate severity package")
        if not monotonic:
            errors.append(f"{key}: non-monotonic severity")

    report = {
        "schema_version": 1,
        "status": "PASSED" if not errors else "FAILED",
        "manifest": str(args.manifest),
        "coordinate_tolerance_m": args.coordinate_tolerance,
        "maximum_coordinate_roundtrip_error_m": max(
            row["coordinate_roundtrip_max_m"] for row in audit_rows
        ),
        "maximum_local_position_abs_m": max(
            row["local_position_max_abs_m"] for row in audit_rows
        ),
        "records": audit_rows,
        "level_checks": level_checks,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"records", "level_checks"}}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
