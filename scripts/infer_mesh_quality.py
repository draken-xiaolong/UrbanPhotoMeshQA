#!/usr/bin/env python3
"""Generate a no-reference quality report for one glTF/GLB/OBJ triangle mesh."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from urbanphotomeshqa.data import pad_mesh_graphs  # noqa: E402
from urbanphotomeshqa.gltf import GltfReader, MeshAsset, sample_surface  # noqa: E402
from urbanphotomeshqa.mesh_attacks import mesh_face_graph, recompute_vertex_normals  # noqa: E402
from urbanphotomeshqa.model import BuildingInvariantEncoder, MeshFaceEncoder  # noqa: E402
from urbanphotomeshqa.morphology import global_morphology_targets  # noqa: E402
from urbanphotomeshqa.patches import patch_descriptors  # noqa: E402
from urbanphotomeshqa.render_features import ImageEncoder, render_views  # noqa: E402

from train_four_branch_fusion import BRANCHES  # noqa: E402
from train_frozen_base_quality_head import ATTACKS  # noqa: E402
from train_no_reference_quality_student import NoReferenceQualityStudent  # noqa: E402
from train_objective_quality_index import GEOMETRY_CLASSES, OrdinalWeightedPatchHead  # noqa: E402


def quality_artifact(name: str) -> Path:
    """Use the organized local layout, with the legacy flat GPU layout as fallback."""
    for group in ("final", "legacy_baseline", "ablations"):
        organized = PROJECT / "artifacts" / "quality" / group / name
        if organized.exists():
            return organized
    return PROJECT / "artifacts" / name


def base_artifact(name: str) -> Path:
    """Use the organized local Base layout, with the legacy flat GPU layout as fallback."""
    organized = PROJECT / "artifacts" / "pretrained_backbone" / name
    return organized if organized.exists() else PROJECT / "artifacts" / name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=base_artifact("four_branch_fusion_seed2026_v1") / "features")
    parser.add_argument("--patch-feature-dir", type=Path, default=quality_artifact("mesh_patch_quality_seed2026_v1"))
    parser.add_argument("--geometry-target-dir", type=Path, default=quality_artifact("objective_geometry_quality_seed2026_v1"))
    parser.add_argument("--texture-target-dir", type=Path, default=quality_artifact("objective_texture_quality_seed2026_v1"))
    parser.add_argument("--quality-student", type=Path, default=quality_artifact("no_reference_patch_quality_seed2026_v3") / "quality_student.pt")
    parser.add_argument("--quality-index", type=Path, default=quality_artifact("objective_quality_index_seed2026_v2") / "quality_index_head.pt")
    parser.add_argument("--quality-index-results", type=Path, default=quality_artifact("objective_quality_index_seed2026_v2") / "results.json")
    parser.add_argument("--point-checkpoint", type=Path, default=base_artifact("formal_invariant_seed2026_v1") / "best.pt")
    parser.add_argument("--mesh-checkpoint", type=Path, default=base_artifact("native_mesh_v1") / "best.pt")
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--patches", type=int, default=16)
    parser.add_argument("--patch-neighbors", type=int, default=32)
    parser.add_argument("--render-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _trimesh_parts(path: Path):
    import trimesh
    loaded = trimesh.load(path, process=False, force="scene")
    if isinstance(loaded, trimesh.Scene):
        return list(loaded.dump(concatenate=False))
    return [loaded]


def load_any_mesh(path: Path) -> tuple[MeshAsset, bool]:
    suffix = path.suffix.lower()
    if suffix == ".gltf":
        asset = GltfReader(path).load_mesh(include_texture=True)
        paths = asset.metadata.get("material_texture_paths", [])
        available = asset.texcoords is not None and np.isfinite(asset.texcoords).any() and any(
            value is not None and Path(value).exists() for value in paths)
        return asset, bool(available)
    if suffix not in {".glb", ".obj"}:
        raise ValueError(f"Unsupported format {suffix}; expected .gltf, .glb or .obj")
    vertices_parts, normals_parts, faces_parts = [], [], []
    materials, texture_arrays, texcoord_parts = [], [], []
    offset = 0
    texture_available = False
    for material_index, mesh in enumerate(_trimesh_parts(path)):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        visual = getattr(mesh, "visual", None)
        uv = getattr(visual, "uv", None)
        image = getattr(getattr(visual, "material", None), "image", None)
        if uv is not None and len(uv) == len(vertices):
            texcoord_parts.append(np.asarray(uv, dtype=np.float64))
        else:
            texcoord_parts.append(np.full((len(vertices), 2), np.nan, dtype=np.float64))
        if image is not None:
            array = np.asarray(image.convert("RGBA") if hasattr(image, "convert") else image, dtype=np.uint8)
            texture_arrays.append(array)
            texture_available = True
        else:
            texture_arrays.append(None)
        vertices_parts.append(vertices); normals_parts.append(normals)
        faces_parts.append(faces + offset); materials.append(np.full(len(faces), material_index, np.int32))
        offset += len(vertices)
    vertices = np.concatenate(vertices_parts)
    faces = np.concatenate(faces_parts)
    normals = np.concatenate(normals_parts)
    if len(normals) != len(vertices) or not np.isfinite(normals).all():
        normals = recompute_vertex_normals(vertices, faces)
    return MeshAsset(
        vertices=vertices, faces=faces, normals=normals,
        face_materials=np.concatenate(materials), texcoords=np.concatenate(texcoord_parts),
        metadata={"asset_id": path.stem, "gltf_path": str(path),
                  "material_texture_arrays": texture_arrays,
                  "vertex_count": len(vertices), "face_count": len(faces)},
    ), texture_available


def feature_statistics(feature_dir: Path):
    with np.load(feature_dir / "scores_train.npz") as values:
        return {branch: (
            values[f"gallery_{branch}"].astype(np.float32).mean(0),
            np.maximum(values[f"gallery_{branch}"].astype(np.float32).std(0), 1e-5),
        ) for branch in BRANCHES}


def patch_statistics(root: Path):
    with np.load(root / "patches_train.npz") as values:
        gallery = values["gallery_patch"][values["gallery_mask"]]
        query = values["query_patch"][values["query_mask"]]
    valid = np.concatenate([gallery, query], 0).astype(np.float32)
    return valid.mean(0), np.maximum(valid.std(0), 1e-5)


def target_statistics(root: Path, gallery_count: int):
    with np.load(root / "targets_train.npz") as values:
        target = values["targets"].astype(np.float32)
        names = values["metrics"].astype(str).tolist()
    target = np.concatenate([np.zeros((gallery_count, target.shape[1]), np.float32), target], 0)
    return target.mean(0), np.maximum(target.std(0), 1e-8), names


def load_encoders(args, device):
    point_state = torch.load(args.point_checkpoint, map_location=device, weights_only=False)
    point = BuildingInvariantEncoder(**point_state["config"]["model"]).to(device).eval()
    point.load_state_dict(point_state["model"])
    mesh_state = torch.load(args.mesh_checkpoint, map_location=device, weights_only=False)
    mesh = MeshFaceEncoder().to(device).eval(); mesh.load_state_dict(mesh_state["model"])
    return point, mesh


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the formal quality report")
    mesh_path = args.mesh.resolve()
    asset, texture_available = load_any_mesh(mesh_path)
    points, sampling = sample_surface(asset, args.points, args.seed)
    face_features, neighbors, topology = mesh_face_graph(asset)
    raw_patches, patch_mask = patch_descriptors(asset, args.patches, args.patch_neighbors)
    point_encoder, mesh_encoder = load_encoders(args, device)
    point_tensor = torch.from_numpy(points[None]).to(device)
    point_feature = F.normalize(point_encoder(point_tensor)["identity"], dim=1)
    graph = pad_mesh_graphs([{"face_features": face_features, "neighbors": neighbors, "topology": topology}])
    graph = {key: value.to(device) for key, value in graph.items()}
    mesh_feature = F.normalize(mesh_encoder(**graph)["identity"], dim=1)
    morphology = global_morphology_targets(point_tensor)
    stats = feature_statistics(args.feature_dir)
    if texture_available:
        texture = torch.from_numpy(ImageEncoder(device)(render_views(asset, args.render_size))["pooled"][None]).to(device)
    else:
        texture = torch.from_numpy(stats["texture"][0][None].astype(np.float32)).to(device)
    raw_branches = [point_feature, mesh_feature, morphology, texture]
    branches = []
    for branch, raw in zip(BRANCHES, raw_branches):
        mean, std = stats[branch]
        branches.append((raw - torch.from_numpy(mean).to(device)) / torch.from_numpy(std).to(device))
    patch_mean, patch_std = patch_statistics(args.patch_feature_dir)
    patch = (raw_patches - patch_mean) / patch_std
    patch[~patch_mask] = 0.0
    patch_tensor = torch.from_numpy(patch[None].astype(np.float32)).to(device)
    patch_mask_tensor = torch.from_numpy(patch_mask[None]).to(device)

    geometry_mean, geometry_std, geometry_names = target_statistics(args.geometry_target_dir, 80)
    texture_mean, texture_std, texture_names = target_statistics(args.texture_target_dir, 80)
    student_state = torch.load(args.quality_student, map_location=device, weights_only=True)
    geometry_only = "geometry_only" in student_state.get("selected_variant", "")
    student = NoReferenceQualityStudent(
        [256, 256, 13, 576], len(geometry_names), len(texture_names), patch_dim=58,
        patch_geometry_only=geometry_only,
        modality_aware=bool(student_state.get("modality_aware", False))).to(device).eval()
    student.load_state_dict(student_state["model"])
    quality = student(
        branches, patch_tensor, patch_mask_tensor,
        texture_available=torch.as_tensor([texture_available], device=device))
    attack_probability = torch.softmax(quality["attack"], 1)
    geometry_raw = torch.clamp(
        quality["geometry"] * torch.from_numpy(geometry_std).to(device)
        + torch.from_numpy(geometry_mean).to(device), min=0.0)
    texture_raw = torch.clamp(
        quality["texture"] * torch.from_numpy(texture_std).to(device)
        + torch.from_numpy(texture_mean).to(device), min=0.0)
    calibration = json.loads(args.quality_index_results.read_text(encoding="utf-8"))
    geometry_scale = torch.as_tensor(calibration["metric_scales_p95"]["geometry"], device=device)
    texture_scale = torch.as_tensor(calibration["metric_scales_p95"]["texture"], device=device)
    geometry_burden = torch.clamp(geometry_raw / geometry_scale, 0.0, 1.0).mean(1)
    texture_burden = torch.clamp(texture_raw / texture_scale, 0.0, 1.0).mean(1)
    geometry_probability = attack_probability[:, list(GEOMETRY_CLASSES)].sum(1)
    texture_probability = torch.clamp(1.0 - geometry_probability - attack_probability[:, 0], min=0.0)
    predicted_burden = geometry_probability * geometry_burden + texture_probability * texture_burden
    deterministic = torch.clamp(1.0 - 0.5 * quality["severity"] - 0.5 * predicted_burden, 0.0, 1.0)
    global_feature = torch.cat([
        quality["quality_embedding"], attack_probability, quality["severity"][:, None],
        quality["geometry"], quality["texture"],
    ], 1)
    index_state = torch.load(args.quality_index, map_location=device, weights_only=True)
    ordinal = OrdinalWeightedPatchHead(global_feature.shape[1], quality["patch_tokens"].shape[2]).to(device).eval()
    ordinal.load_state_dict(index_state["model"])
    ordinal_score, ordinal_aux = ordinal(global_feature, quality["patch_tokens"], patch_mask_tensor)
    ordinal_weight = float(index_state.get("ordinal_weight", 0.75))
    full_overall = ordinal_weight * ordinal_score + (1.0 - ordinal_weight) * deterministic
    geometry_only = torch.clamp(1.0 - 0.5 * quality["severity"] - 0.5 * geometry_burden, 0.0, 1.0)
    overall = full_overall if texture_available else geometry_only

    local_logits = ordinal.patch_head(quality["patch_tokens"])
    local_scores = torch.sigmoid(local_logits).mean(2)[0].cpu().numpy()
    reliability = ordinal_aux["patch_weights"][0].cpu().numpy()
    risk = (1.0 - local_scores) * reliability
    valid_indices = np.flatnonzero(patch_mask)
    worst = valid_indices[np.argsort(-risk[valid_indices])[: min(3, len(valid_indices))]]
    predicted_class = int(attack_probability.argmax(1).item())
    report = {
        "schema_version": 1, "model_path": str(mesh_path), "format": mesh_path.suffix.lower()[1:],
        "objective_quality_index": round(float(overall.item() * 100.0), 3),
        "quality_scope": ("geometry_and_texture" if texture_available else "geometry_only"),
        "geometry_quality": round(float((1.0 - geometry_burden).item() * 100.0), 3),
        "texture_quality": (round(float((1.0 - texture_burden).item() * 100.0), 3)
                            if texture_available else None),
        "predicted_degradation": ATTACKS[predicted_class],
        "degradation_confidence": round(float(attack_probability[0, predicted_class]), 4),
        "predicted_severity": round(float(quality["severity"].item()), 4),
        "texture_available": texture_available,
        "low_quality_patches": [{
            "patch_index": int(index), "normalized_center": raw_patches[index, :3].round(6).tolist(),
            "local_quality": round(float(local_scores[index] * 100.0), 3),
            "reliability_weight": round(float(reliability[index]), 6),
        } for index in worst],
        "mesh_audit": {
            "vertices": int(len(asset.vertices)), "faces": int(len(asset.faces)),
            "surface_area": float(sampling["surface_area"]),
            "degenerate_faces": int(sampling["degenerate_face_count"]),
            "sample_points": args.points, "patches": int(patch_mask.sum()),
        },
        "component_predictions": {
            "geometry": {name: float(value) for name, value in zip(geometry_names, geometry_raw[0].cpu())},
            "texture": ({name: float(value) for name, value in zip(texture_names, texture_raw[0].cpu())}
                        if texture_available else None),
            "attack_probabilities": {name: round(float(value), 6)
                                     for name, value in zip(ATTACKS, attack_probability[0].cpu())},
        },
        "protocol": {
            "no_reference": True, "seed": args.seed, "higher_is_better": True,
            "score_is_mos": False, "warning": (None if texture_available else
                "No usable texture was found; OQI is geometry-only and texture quality is not inferred."),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "model_path", "format", "objective_quality_index", "quality_scope", "geometry_quality", "texture_quality",
            "predicted_degradation", "degradation_confidence", "predicted_severity", "texture_available"])
        writer.writeheader(); writer.writerow({key: report[key] for key in writer.fieldnames})
    print(json.dumps({key: report[key] for key in (
        "model_path", "objective_quality_index", "geometry_quality", "texture_quality",
        "predicted_degradation", "degradation_confidence", "predicted_severity")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
