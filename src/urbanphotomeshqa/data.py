from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_record_paths(manifest_path: str | Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = Path(manifest_path).resolve().parent
    resolved = []
    for record in records:
        item = dict(record)
        cache_path = Path(item["cache_path"])
        item["cache_path"] = str(cache_path if cache_path.is_absolute() else base / cache_path)
        resolved.append(item)
    return resolved


def rotation_z(points: np.ndarray, angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    result = points.copy()
    result[:, :3] = result[:, :3] @ matrix.T
    result[:, 3:6] = result[:, 3:6] @ matrix.T
    return result


def degrade_points(points: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, str]:
    """Return an identity-preserving view and a synthetic degradation severity in [0, 1]."""
    result = points.copy()
    result = rotation_z(result, float(rng.uniform(-math.pi, math.pi)))
    result[:, :3] *= float(rng.uniform(0.85, 1.15))
    result[:, :3] += rng.uniform(-0.1, 0.1, size=(1, 3)).astype(np.float32)

    attack = str(rng.choice(["clean", "noise", "dropout", "quantize", "crop"]))
    if attack == "clean":
        severity = 0.0
    else:
        severity = float(rng.uniform(0.1, 0.8))
    if attack == "noise":
        result[:, :3] += rng.normal(0.0, 0.0125 * severity, size=result[:, :3].shape).astype(np.float32)
    elif attack == "dropout":
        keep = max(32, int(len(result) * (1.0 - 0.65 * severity)))
        chosen = rng.choice(len(result), size=keep, replace=False)
        result = result[chosen]
    elif attack == "quantize":
        step = 0.002 + 0.035 * severity
        result[:, :3] = np.round(result[:, :3] / step) * step
    elif attack == "crop":
        axis = int(rng.integers(0, 3))
        threshold = np.quantile(result[:, axis], min(0.45, 0.08 + 0.4 * severity))
        result = result[result[:, axis] >= threshold]

    target_count = len(points)
    selected = rng.choice(len(result), size=target_count, replace=len(result) < target_count)
    result = result[selected]
    rng.shuffle(result)
    return result.astype(np.float32), severity, attack


def local_correspondence_view(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply identity-preserving transforms without changing point correspondence/order."""
    result = rotation_z(points, float(rng.uniform(-math.pi, math.pi)))
    result[:, :3] *= float(rng.uniform(0.7, 1.25))
    result[:, :3] += rng.uniform(-0.15, 0.15, size=(1, 3)).astype(np.float32)
    jitter = float(rng.uniform(0.0, 0.004))
    result[:, :3] += rng.normal(0.0, jitter, size=result[:, :3].shape).astype(np.float32)
    return result.astype(np.float32)


def deterministic_split(asset_ids: list[str], seed: int, fractions: dict[str, float]) -> dict[str, list[str]]:
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Split fractions must sum to 1")
    ordered = np.array(sorted(asset_ids), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    train_end = int(len(ordered) * fractions["train"])
    val_end = train_end + int(len(ordered) * fractions["val"])
    return {
        "train": ordered[:train_end].tolist(),
        "val": ordered[train_end:val_end].tolist(),
        "test": ordered[val_end:].tolist(),
    }


class BuildingPairDataset:
    def __init__(self, manifest_path: str | Path, split: str, seed: int = 2026):
        manifest = load_manifest(manifest_path)
        wanted = set(manifest["splits"][split])
        records = resolve_record_paths(manifest_path, manifest["records"])
        self.records = [record for record in records if record["asset_id"] in wanted]
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int):
        import torch

        record = self.records[index]
        points = np.load(record["cache_path"])["points"].astype(np.float32)
        base_seed = self.seed + self.epoch * 1_000_003 + index * 97
        view_a, quality_a, attack_a = degrade_points(points, np.random.default_rng(base_seed))
        view_b, quality_b, attack_b = degrade_points(points, np.random.default_rng(base_seed + 1))
        local_view = local_correspondence_view(points, np.random.default_rng(base_seed + 2))
        return {
            "reference": torch.from_numpy(points.copy()),
            "local_view": torch.from_numpy(local_view),
            "view_a": torch.from_numpy(view_a),
            "view_b": torch.from_numpy(view_b),
            "quality_a": torch.tensor(quality_a, dtype=torch.float32),
            "quality_b": torch.tensor(quality_b, dtype=torch.float32),
            "asset_id": record["asset_id"],
            "attack_a": attack_a,
            "attack_b": attack_b,
        }


class BuildingEvalDataset:
    def __init__(self, manifest_path: str | Path, split: str, seed: int = 2026):
        manifest = load_manifest(manifest_path)
        wanted = set(manifest["splits"][split])
        records = resolve_record_paths(manifest_path, manifest["records"])
        self.records = [record for record in records if record["asset_id"] in wanted]
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        record = self.records[index]
        points = np.load(record["cache_path"])["points"].astype(np.float32)
        query, severity, attack = degrade_points(points, np.random.default_rng(self.seed + index * 131))
        return {
            "gallery": torch.from_numpy(points),
            "query": torch.from_numpy(query),
            "quality": torch.tensor(severity, dtype=torch.float32),
            "asset_id": record["asset_id"],
            "attack": attack,
        }


class MeshAttackPairDataset:
    """One canonical mesh view and two genuine mesh-attack views per building."""

    def __init__(self, attack_manifest_path: str | Path, seed: int = 2026):
        manifest_path = Path(attack_manifest_path).resolve()
        manifest = load_manifest(manifest_path)
        base = manifest_path.parent
        self.galleries = {
            row["asset_id"]: str(base / row["cache_path"]) for row in manifest["galleries"]
        }
        self.variants: dict[str, list[dict[str, Any]]] = {asset_id: [] for asset_id in self.galleries}
        for row in manifest["records"]:
            if row["attack"] == "clean_resample":
                continue
            item = dict(row)
            item["cache_path"] = str(base / item["cache_path"])
            self.variants[item["asset_id"]].append(item)
        self.asset_ids = sorted(asset_id for asset_id, rows in self.variants.items() if rows)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.asset_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int):
        import torch

        asset_id = self.asset_ids[index]
        variants = self.variants[asset_id]
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 101)
        selected = rng.choice(len(variants), size=2, replace=len(variants) < 2)
        first, second = variants[int(selected[0])], variants[int(selected[1])]
        reference = np.load(self.galleries[asset_id])["points"].astype(np.float32)
        attack_a = np.load(first["cache_path"])["points"].astype(np.float32)
        attack_b = np.load(second["cache_path"])["points"].astype(np.float32)
        return {
            "reference": torch.from_numpy(reference),
            "attack_a": torch.from_numpy(attack_a),
            "attack_b": torch.from_numpy(attack_b),
            "asset_id": asset_id,
            "attack_name_a": first["attack"],
            "attack_name_b": second["attack"],
        }


def _load_mesh_graph(path: str | Path) -> dict[str, Any]:
    values = np.load(path)
    required = {"face_features", "neighbors", "topology"}
    if not required.issubset(values.files):
        raise ValueError(f"Mesh graph arrays missing from {path}; rerun preparation with --store-mesh-graphs")
    return {
        "face_features": values["face_features"].astype(np.float32),
        "neighbors": values["neighbors"].astype(np.int64),
        "topology": values["topology"].astype(np.float32),
    }


class MeshAttackGraphPairDataset(MeshAttackPairDataset):
    """Native face-graph counterpart of MeshAttackPairDataset."""

    def __getitem__(self, index: int):
        asset_id = self.asset_ids[index]
        variants = self.variants[asset_id]
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 101)
        selected = rng.choice(len(variants), size=2, replace=len(variants) < 2)
        first, second = variants[int(selected[0])], variants[int(selected[1])]
        return {
            "reference": _load_mesh_graph(self.galleries[asset_id]),
            "attack_a": _load_mesh_graph(first["cache_path"]),
            "attack_b": _load_mesh_graph(second["cache_path"]),
            "asset_id": asset_id,
        }


def pad_mesh_graphs(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    maximum = max(len(graph["face_features"]) for graph in graphs)
    feature_dim = graphs[0]["face_features"].shape[1]
    features = np.zeros((len(graphs), maximum, feature_dim), dtype=np.float32)
    neighbors = np.zeros((len(graphs), maximum, 3), dtype=np.int64)
    mask = np.zeros((len(graphs), maximum), dtype=bool)
    topology = np.stack([graph["topology"] for graph in graphs])
    for index, graph in enumerate(graphs):
        count = len(graph["face_features"])
        features[index, :count] = graph["face_features"]
        neighbors[index, :count] = graph["neighbors"]
        mask[index, :count] = True
    return {
        "face_features": torch.from_numpy(features),
        "neighbors": torch.from_numpy(neighbors),
        "mask": torch.from_numpy(mask),
        "topology": torch.from_numpy(topology),
    }


def collate_mesh_graph_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reference": pad_mesh_graphs([item["reference"] for item in items]),
        "attack_a": pad_mesh_graphs([item["attack_a"] for item in items]),
        "attack_b": pad_mesh_graphs([item["attack_b"] for item in items]),
        "asset_id": [item["asset_id"] for item in items],
    }
