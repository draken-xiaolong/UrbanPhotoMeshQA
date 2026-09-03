#!/usr/bin/env python3
"""Build clean-Patch-aligned geometry targets from exact triangle surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.patches import topological_patch_layout


ATTACKS = {"geometry_hole", "mesh_simplification_qem", "geometry_noise_spike"}
METRICS = ("symmetric_chamfer_l1", "symmetric_distance_p99", "symmetric_hausdorff",
           "symmetric_normal_error", "clean_missing_fraction_0p005",
           "clean_missing_fraction_0p01")
LEVEL = {"light": 0, "medium": 1, "heavy": 2}


def stable_seed(seed, *parts):
    return int.from_bytes(hashlib.sha256("|".join((str(seed), *parts)).encode()).digest()[:8], "little")


def frame(mesh):
    minimum = mesh.vertices.min(0); maximum = mesh.vertices.max(0)
    return .5 * (minimum + maximum), max(float(np.linalg.norm(maximum - minimum)), 1e-12)


def sample(mesh, center, scale, count, seed, face_subset=None):
    vertices = (np.asarray(mesh.vertices, np.float64) - center) / scale
    face_ids = (np.arange(len(mesh.faces)) if face_subset is None else np.asarray(face_subset))
    triangles = vertices[np.asarray(mesh.faces, np.int64)[face_ids]]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1); valid = areas > 1e-14
    face_ids = face_ids[valid]; triangles = triangles[valid]; cross = cross[valid]; areas = areas[valid]
    if not len(face_ids): raise ValueError("No non-degenerate faces to sample")
    rng = np.random.default_rng(seed); chosen = rng.choice(len(face_ids), count, p=areas / areas.sum())
    r1 = np.sqrt(rng.random((count, 1))); r2 = rng.random((count, 1))
    bary = np.concatenate([1-r1, r1*(1-r2), r1*r2], 1)
    points = np.sum(triangles[chosen] * bary[:, :, None], 1)
    normals = cross[chosen] / areas[chosen, None]
    return points.astype(np.float32), normals.astype(np.float32), face_ids[chosen]


def scene(mesh, center, scale):
    import open3d as o3d
    vertices = ((np.asarray(mesh.vertices, np.float64) - center) / scale).astype(np.float32)
    faces = np.asarray(mesh.faces, np.uint32); triangles = vertices[faces]
    valid = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                    triangles[:, 2]-triangles[:, 0]), axis=1) > 1e-12
    face_ids = np.flatnonzero(valid); faces = faces[valid]
    tensor_mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
    result = o3d.t.geometry.RaycastingScene(); result.add_triangles(tensor_mesh)
    return result, face_ids


def closest(target_scene, points):
    import open3d as o3d
    query = target_scene.compute_closest_points(o3d.core.Tensor(points))
    target = query["points"].numpy(); normals = query["primitive_normals"].numpy()
    return np.linalg.norm(points-target, axis=1), normals, query["primitive_ids"].numpy()


def prepare_clean(mesh, face_patch, samples_per_patch, seed):
    center, scale = frame(mesh); points=[]; normals=[]; labels=[]
    for patch in range(int(face_patch.max())+1):
        p, n, _ = sample(mesh, center, scale, samples_per_patch,
                         stable_seed(seed, "clean", str(patch)), np.flatnonzero(face_patch == patch))
        points.append(p); normals.append(n); labels.append(np.full(len(p), patch, np.int64))
    return center, scale, np.concatenate(points), np.concatenate(normals), np.concatenate(labels), scene(mesh, center, scale)


def local_metrics(clean, degraded, samples_per_patch, seed):
    center, scale, clean_points, clean_normals, clean_labels, (clean_scene, clean_scene_faces) = clean
    layout = topological_patch_layout(degraded, 16)
    degraded_face_patch = layout["face_patch"].astype(np.int64)
    active = int(layout["patch_mask"].sum())
    degraded_scene, degraded_scene_faces = scene(degraded, center, scale)
    degraded_points=[]; degraded_normals=[]; degraded_labels=[]
    for patch in range(active):
        points,normals,_=sample(degraded,center,scale,samples_per_patch,
            stable_seed(seed,"degraded",str(patch)),np.flatnonzero(degraded_face_patch==patch))
        degraded_points.append(points); degraded_normals.append(normals)
        degraded_labels.append(np.full(samples_per_patch,patch,np.int64))
    degraded_points=np.concatenate(degraded_points); degraded_normals=np.concatenate(degraded_normals)
    degraded_labels=np.concatenate(degraded_labels)
    c2d, degraded_nearest_normals, degraded_primitive = closest(degraded_scene, clean_points)
    d2c, clean_nearest_normals, primitive = closest(clean_scene, degraded_points)
    c2d[c2d < 1e-7] = 0.; d2c[d2c < 1e-7] = 0.
    clean_assigned_labels = degraded_face_patch[degraded_scene_faces[degraded_primitive]]
    c_normal = 1 - np.clip(np.abs(np.sum(clean_normals * degraded_nearest_normals, 1)), 0, 1)
    d_normal = 1 - np.clip(np.abs(np.sum(degraded_normals * clean_nearest_normals, 1)), 0, 1)
    c_normal[c_normal < 1e-6] = 0.; d_normal[d_normal < 1e-6] = 0.
    values=np.zeros((16,len(METRICS)),np.float32); sample_counts=np.zeros((16,2),np.int64)
    patch_mask=np.zeros(16,bool); patch_mask[:active]=True
    for patch in range(active):
        ci = clean_assigned_labels == patch; di = degraded_labels == patch
        sample_counts[patch] = [ci.sum(), di.sum()]
        forward_distance = c2d[ci] if ci.any() else d2c[di]
        forward_normal = c_normal[ci] if ci.any() else d_normal[di]
        distances = np.concatenate([forward_distance, d2c[di]])
        normals = np.concatenate([forward_normal, d_normal[di]])
        values[patch] = [.5*(forward_distance.mean()+d2c[di].mean()), np.quantile(distances,.99),
                         distances.max(initial=0), normals.mean(),
                         np.mean(forward_distance>.005), np.mean(forward_distance>.01)]
    return values, sample_counts, patch_mask


def quality(values, config):
    index={name:i for i,name in enumerate(METRICS)}; output=np.ones(len(values),np.float32)
    for patch,row in enumerate(values):
        scores=[]
        for component in ("geometry_fidelity","completeness"):
            burden=sum(float(setting["weight"])*max(float(row[index[name]]),0)/float(setting["scale"])
                       for name,setting in config[component].items())
            scores.append(np.exp(-burden))
        output[patch]=min(scores)
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest",type=Path,required=True)
    parser.add_argument("--patch-map-dir",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--samples-per-patch",type=int,default=1024)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--asset-ids",nargs="*")
    parser.add_argument("--shard-index",type=int,default=0); parser.add_argument("--shard-count",type=int,default=1)
    args=parser.parse_args(); manifest=json.loads(args.dataset_manifest.read_text())["records"]
    requested=set(args.asset_ids or []); eligible=[r for r in manifest if r["attack"] in ATTACKS|{"clean"}
                                                   and (not requested or r["asset_id"] in requested)]
    assets=sorted({r["asset_id"] for r in eligible}); shard=set(assets[args.shard_index::args.shard_count])
    records=[r for r in eligible if r["asset_id"] in shard]; config=json.loads(args.config.read_text())
    rows=[]
    for asset_index,asset_id in enumerate(sorted(shard)):
        asset_rows=[r for r in records if r["asset_id"]==asset_id]
        clean_row=next(r for r in asset_rows if r["attack"]=="clean")
        clean_mesh=GltfReader(clean_row["gltf_path"]).load_mesh(include_texture=False)
        with np.load(args.patch_map_dir/f"{asset_id}.npz") as sidecar: face_patch=sidecar["face_patch"].astype(np.int64)
        prepared=prepare_clean(clean_mesh,face_patch,args.samples_per_patch,stable_seed(args.seed,asset_id))
        for record in asset_rows:
            if record["attack"]=="clean":
                metric=np.zeros((16,len(METRICS)),np.float32); counts=np.full((16,2),args.samples_per_patch)
                patch_mask=np.ones(16,bool)
            else:
                degraded=GltfReader(record["gltf_path"]).load_mesh(include_texture=False)
                metric,counts,patch_mask=local_metrics(prepared,degraded,args.samples_per_patch,
                    stable_seed(args.seed,asset_id,record["attack"],record["level"]))
            raw=quality(metric,config); noop=np.max(metric,axis=1)<=1e-12
            rows.append({"record":record,"metrics":metric,"raw":raw,"quality":raw.copy(),
                         "counts":counts,"noop":noop,"patch_mask":patch_mask})
        print(f"asset {asset_index+1}/{len(shard)} {asset_id}",flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for split in ("train","val","test","blind"):
        selected=[r for r in rows if r["record"]["split"]==split]; shape=(0,16)
        np.savez_compressed(args.output_dir/f"patch_geometry_targets_{split}.npz",
            asset_ids=np.asarray([r["record"]["asset_id"] for r in selected]), attacks=np.asarray([r["record"]["attack"] for r in selected]),
            levels=np.asarray([r["record"]["level"] for r in selected]), metric_names=np.asarray(METRICS),
            patch_geometry_quality=np.stack([r["quality"] for r in selected]) if selected else np.empty(shape),
            patch_geometry_quality_raw=np.stack([r["raw"] for r in selected]) if selected else np.empty(shape),
            patch_metrics=np.stack([r["metrics"] for r in selected]) if selected else np.empty((*shape,len(METRICS))),
            directional_sample_count=np.stack([r["counts"] for r in selected]) if selected else np.empty((*shape,2),np.int64),
            objective_noop=np.stack([r["noop"] for r in selected]) if selected else np.empty(shape,bool),
            patch_mask=np.stack([r["patch_mask"] for r in selected]) if selected else np.empty(shape,bool))
    metadata={"schema_version":1,"seed":args.seed,"assets":len(shard),"records":len(rows),"patches":16,
              "samples_per_clean_patch":args.samples_per_patch,"metric_names":METRICS,"shard_index":args.shard_index,
              "shard_count":args.shard_count,"local_quality":"minimum of geometry_fidelity and completeness; no fabricated local topology term",
              "patch_alignment":"each attacked mesh is partitioned independently; clean samples inherit the nearest attacked face Patch",
              "supervision":"full-reference offline Patch geometry targets; inference remains no-reference"}
    (args.output_dir/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n"); print(json.dumps({"status":"COMPLETE",**metadata}))


if __name__=="__main__": main()
