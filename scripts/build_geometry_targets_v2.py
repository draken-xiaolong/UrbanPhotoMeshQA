#!/usr/bin/env python3
"""Build dense point-to-triangle geometry targets in the clean coordinate frame."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.mesh_attacks import topology_stats


GEOMETRY_ATTACKS = {
    "geometry_hole",
    "mesh_simplification_qem",
    "geometry_noise_spike",
}
METRIC_NAMES = (
    "symmetric_chamfer_l1",
    "symmetric_chamfer_l2",
    "symmetric_distance_p95",
    "symmetric_distance_p99",
    "symmetric_hausdorff",
    "symmetric_normal_error",
    "clean_missing_fraction_0p0025",
    "clean_missing_fraction_0p005",
    "clean_missing_fraction_0p01",
    "clean_missing_fraction_0p02",
    "degraded_excess_fraction_0p005",
    "face_ratio_error",
    "surface_area_ratio_error",
    "boundary_edge_ratio_delta",
    "nonmanifold_edge_ratio_delta",
    "component_log_delta",
)


def stable_seed(seed: int, *parts: str) -> int:
    value = "|".join((str(seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def clean_frame(mesh):
    minimum = mesh.vertices.min(axis=0)
    maximum = mesh.vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    scale = max(float(np.linalg.norm(maximum - minimum)), 1e-12)
    return center, scale


def sample_surface_in_frame(mesh, center, scale, count, seed):
    rng = np.random.default_rng(seed)
    vertices = (np.asarray(mesh.vertices, np.float64) - center) / scale
    triangles = vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    probabilities = np.where(double_area > 1e-14, double_area, 0.0)
    if probabilities.sum() <= 0:
        raise ValueError("Mesh has no non-degenerate faces")
    probabilities /= probabilities.sum()
    selected = rng.choice(len(triangles), count, replace=True, p=probabilities)
    r1 = np.sqrt(rng.random((count, 1)))
    r2 = rng.random((count, 1))
    barycentric = np.concatenate(
        [1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1
    )
    points = np.sum(triangles[selected] * barycentric[:, :, None], axis=1)
    normals = cross[selected] / np.maximum(double_area[selected, None], 1e-14)
    return points.astype(np.float32), normals.astype(np.float32)


def raycasting_scene(mesh, center, scale):
    import open3d as o3d

    vertices = ((np.asarray(mesh.vertices, np.float64) - center) / scale).astype(np.float32)
    faces = np.asarray(mesh.faces, np.uint32)
    triangles = vertices[faces]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    faces = faces[double_area > 1e-12]
    if not len(faces):
        raise ValueError("Mesh has no non-degenerate faces for ray casting")
    triangle_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices), o3d.core.Tensor(faces)
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(triangle_mesh)
    return scene


def closest(scene, points):
    import open3d as o3d

    query = scene.compute_closest_points(o3d.core.Tensor(points))
    closest_points = query["points"].numpy()
    normals = query["primitive_normals"].numpy()
    distances = np.linalg.norm(points - closest_points, axis=1)
    return distances, normals


def ratio_delta(clean_value, degraded_value):
    return abs(float(degraded_value) / max(float(clean_value), 1e-12) - 1.0)


def geometry_metrics(clean, degraded, sample_count, seed):
    center, scale = clean_frame(clean)
    clean_points, clean_normals = sample_surface_in_frame(
        clean, center, scale, sample_count, stable_seed(seed, "clean")
    )
    degraded_points, degraded_normals = sample_surface_in_frame(
        degraded, center, scale, sample_count, stable_seed(seed, "degraded")
    )
    clean_scene = raycasting_scene(clean, center, scale)
    degraded_scene = raycasting_scene(degraded, center, scale)
    clean_to_degraded, degraded_face_normals = closest(degraded_scene, clean_points)
    degraded_to_clean, clean_face_normals = closest(clean_scene, degraded_points)
    distances = np.concatenate([clean_to_degraded, degraded_to_clean])
    normal_error = 0.5 * (
        np.mean(1.0 - np.clip(
            np.abs(np.sum(clean_normals * degraded_face_normals, axis=1)), 0.0, 1.0))
        + np.mean(1.0 - np.clip(
            np.abs(np.sum(degraded_normals * clean_face_normals, axis=1)), 0.0, 1.0))
    )

    clean_topology = topology_stats(clean)
    degraded_topology = topology_stats(degraded)
    clean_edges = max(clean_topology["edge_count"], 1)
    degraded_edges = max(degraded_topology["edge_count"], 1)
    clean_boundary = clean_topology["boundary_edge_count"] / clean_edges
    degraded_boundary = degraded_topology["boundary_edge_count"] / degraded_edges
    clean_nonmanifold = clean_topology["nonmanifold_edge_count"] / clean_edges
    degraded_nonmanifold = degraded_topology["nonmanifold_edge_count"] / degraded_edges
    values = np.asarray([
        0.5 * (clean_to_degraded.mean() + degraded_to_clean.mean()),
        0.5 * (np.mean(clean_to_degraded ** 2) + np.mean(degraded_to_clean ** 2)),
        np.quantile(distances, 0.95),
        np.quantile(distances, 0.99),
        distances.max(initial=0.0),
        normal_error,
        np.mean(clean_to_degraded > 0.0025),
        np.mean(clean_to_degraded > 0.005),
        np.mean(clean_to_degraded > 0.01),
        np.mean(clean_to_degraded > 0.02),
        np.mean(degraded_to_clean > 0.005),
        ratio_delta(clean_topology["face_count"], degraded_topology["face_count"]),
        ratio_delta(clean_topology["surface_area"], degraded_topology["surface_area"]),
        abs(clean_boundary - degraded_boundary),
        abs(clean_nonmanifold - degraded_nonmanifold),
        abs(np.log1p(clean_topology["connected_components"])
            - np.log1p(degraded_topology["connected_components"])),
    ], dtype=np.float32)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--asset-ids", nargs="*")
    args = parser.parse_args()

    payload = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    requested = set(args.asset_ids or [])
    records = [
        row for row in payload["records"]
        if row["attack"] in GEOMETRY_ATTACKS | {"clean"}
        and (not requested or row["asset_id"] in requested)
    ]
    if requested and requested != {row["asset_id"] for row in records}:
        raise ValueError(f"Unknown requested assets: {sorted(requested - {row['asset_id'] for row in records})}")
    clean_paths = {
        row["asset_id"]: row["gltf_path"] for row in records if row["attack"] == "clean"
    }
    clean_meshes = {
        asset_id: GltfReader(path).load_mesh(include_texture=False)
        for asset_id, path in clean_paths.items()
    }
    rows = []
    for index, record in enumerate(records):
        if record["attack"] == "clean":
            metrics = np.zeros(len(METRIC_NAMES), np.float32)
        else:
            degraded = GltfReader(record["gltf_path"]).load_mesh(include_texture=False)
            metrics = geometry_metrics(
                clean_meshes[record["asset_id"]],
                degraded,
                args.samples,
                stable_seed(args.seed, record["asset_id"], record["attack"], record["level"]),
            )
        rows.append((record, metrics))
        print(f"target {index + 1}/{len(records)} {record['asset_id']} "
              f"{record['attack']} {record['level']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test", "blind"):
        selected = [row for row in rows if row[0]["split"] == split]
        metric_values = (
            np.stack([row[1] for row in selected])
            if selected else np.empty((0, len(METRIC_NAMES)), dtype=np.float32)
        )
        np.savez_compressed(
            args.output_dir / f"geometry_targets_v2_{split}.npz",
            asset_ids=np.asarray([row[0]["asset_id"] for row in selected]),
            attacks=np.asarray([row[0]["attack"] for row in selected]),
            levels=np.asarray([row[0]["level"] for row in selected]),
            metrics=metric_values,
            metric_names=np.asarray(METRIC_NAMES),
        )
    metadata = {
        "schema_version": 2,
        "method": "deterministic dense surface samples queried against exact triangle surfaces",
        "coordinate_frame": "clean bbox center and diagonal for both clean and degraded meshes",
        "sample_count_per_direction": args.samples,
        "seed": args.seed,
        "metric_names": METRIC_NAMES,
        "scope": "pilot" if requested else "formal",
        "assets": len(clean_meshes),
        "records": len(rows),
        "dataset_manifest": str(args.dataset_manifest),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETE", **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
