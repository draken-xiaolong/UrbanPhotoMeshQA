from __future__ import annotations

from typing import Any

import numpy as np


ATTACKS = ("clean", "noise", "dropout", "quantize", "crop")


def canonical_degradation(
    points: np.ndarray,
    attack: str,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a quality degradation in canonical coordinates, before benign pose changes."""
    if attack not in ATTACKS:
        raise ValueError(f"Unknown attack: {attack}")
    result = points.copy()
    severity = float(np.clip(severity, 0.0, 1.0))
    if attack == "noise":
        result[:, :3] += rng.normal(
            0.0, 0.0125 * severity, size=result[:, :3].shape
        ).astype(np.float32)
    elif attack == "dropout":
        keep = max(32, int(len(result) * (1.0 - 0.65 * severity)))
        result = result[rng.choice(len(result), size=keep, replace=False)]
    elif attack == "quantize":
        step = 0.002 + 0.035 * severity
        result[:, :3] = np.round(result[:, :3] / step) * step
    elif attack == "crop":
        axis = int(rng.integers(0, 3))
        threshold = np.quantile(result[:, axis], min(0.45, 0.08 + 0.4 * severity))
        result = result[result[:, axis] >= threshold]
    return result.astype(np.float32)


def fixed_count(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    selected = rng.choice(len(points), size=count, replace=len(points) < count)
    result = points[selected].copy()
    rng.shuffle(result)
    return result.astype(np.float32)


def nearest(reference_xyz: np.ndarray, query_xyz: np.ndarray, chunk: int = 256):
    distances, indices = [], []
    for start in range(0, len(reference_xyz), chunk):
        block = reference_xyz[start : start + chunk]
        squared = np.sum((block[:, None, :] - query_xyz[None, :, :]) ** 2, axis=2)
        best = np.argmin(squared, axis=1)
        distances.append(np.sqrt(squared[np.arange(len(block)), best]))
        indices.append(best)
    return np.concatenate(distances), np.concatenate(indices)


def objective_quality_metrics(
    reference: np.ndarray,
    degraded: np.ndarray,
    completeness_threshold: float = 0.015,
) -> dict[str, Any]:
    ref_distance, ref_match = nearest(reference[:, :3], degraded[:, :3])
    deg_distance, deg_match = nearest(degraded[:, :3], reference[:, :3])
    ref_normals = reference[:, 3:6]
    deg_normals = degraded[:, 3:6]
    ref_normal_dot = np.abs(np.sum(ref_normals * deg_normals[ref_match], axis=1))
    deg_normal_dot = np.abs(np.sum(deg_normals * ref_normals[deg_match], axis=1))
    ref_extent = np.ptp(reference[:, :3], axis=0)
    degraded_extent = np.ptp(degraded[:, :3], axis=0)
    # A diagonal-normalized vector error remains defined for intrinsically thin
    # objects such as water surfaces; per-axis relative error explodes when an
    # object's thickness is legitimately close to zero.
    extent_error = np.linalg.norm(degraded_extent - ref_extent) / max(
        np.linalg.norm(ref_extent), 1e-6
    )
    return {
        "chamfer_l2": float(0.5 * (np.mean(ref_distance**2) + np.mean(deg_distance**2))),
        "hausdorff": float(max(ref_distance.max(), deg_distance.max())),
        "normal_error": float(1.0 - 0.5 * (ref_normal_dot.mean() + deg_normal_dot.mean())),
        "missing_fraction": float(np.mean(ref_distance > completeness_threshold)),
        "outlier_fraction": float(np.mean(deg_distance > completeness_threshold)),
        "bbox_extent_relative_error": float(extent_error),
        "reference_points": int(len(reference)),
        "degraded_points": int(len(degraded)),
    }
