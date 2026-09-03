#!/usr/bin/env python3
"""Train and ablate a frozen-Base no-reference quality head on real glTF packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ATTACKS = ("clean", "geometry_hole", "mesh_simplification_qem", "geometry_noise_spike",
           "texture_detail_loss", "texture_region_missing", "texture_misalignment")
BRANCHES = ("point", "mesh", "morphology", "texture")
VARIANTS = {
    "point": ((0,), False),
    "point_mesh_morphology": ((0, 1, 2), False),
    "four_branch": ((0, 1, 2, 3), False),
    "four_branch_patch": ((0, 1, 2, 3), True),
}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rankdata(values):
    """Return average ranks so ties receive the same Spearman rank."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else 0.0


def macro_f1(predicted, truth):
    scores = []
    for label in range(len(ATTACKS)):
        tp = np.sum((predicted == label) & (truth == label))
        fp = np.sum((predicted == label) & (truth != label))
        fn = np.sum((predicted != label) & (truth == label))
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores)), scores


FORMAL_COUNTS = {"train": 1518, "val": 760, "test": 607, "blind": 608}


def dataset_provenance(manifest_path, raw, require_formal=False):
    if manifest_path is None:
        if require_formal:
            raise ValueError("--require-formal requires --dataset-manifest")
        return None
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    ordered = {}
    for split in ("train", "val", "test", "blind"):
        manifest_rows = [row for row in records if row["split"] == split]
        expected = [(str(row["asset_id"]), str(row["attack"]), str(row["level"]))
                    for row in manifest_rows]
        if split in raw:
            observed = list(zip(raw[split]["asset_ids"].astype(str),
                                raw[split]["attacks"].astype(str),
                                raw[split]["levels"].astype(str)))
            if expected != observed:
                raise ValueError(f"Dataset manifest order mismatch: {split}")
        ordered[split] = expected
    counts = {split: len(rows) for split, rows in ordered.items()}
    exclusions = payload.get("quality_control", {}).get(
        "excluded_exact_duplicate_severity_packages", 0)
    if require_formal and (counts != FORMAL_COUNTS or exclusions != 3):
        raise ValueError(
            f"Formal dataset required; got counts={counts}, exclusions={exclusions}"
        )
    canonical = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    return {
        "manifest": str(Path(manifest_path)),
        "ordered_sample_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "counts": counts,
        "excluded_exact_duplicate_severity_packages": int(exclusions),
        "formal": bool(counts == FORMAL_COUNTS and exclusions == 3),
    }


class Store:
    def __init__(self, root, device, objective_target_dir=None, dataset_manifest=None,
                 require_formal=False, normalization="mean_std",
                 splits=("train", "val", "test", "blind")):
        self.splits = tuple(splits)
        if not {"train", "val"}.issubset(self.splits):
            raise ValueError("Store requires train and val splits")
        raw = {}
        for split in self.splits:
            with np.load(root / f"features_{split}.npz") as values:
                raw[split] = {name: values[name].copy() for name in values.files}
            if objective_target_dir is not None:
                with np.load(objective_target_dir / f"objective_targets_{split}.npz") as objective:
                    if not (np.array_equal(raw[split]["asset_ids"].astype(str), objective["asset_ids"].astype(str))
                            and np.array_equal(raw[split]["attacks"].astype(str), objective["attacks"].astype(str))
                            and np.array_equal(raw[split]["levels"].astype(str), objective["levels"].astype(str))):
                        raise ValueError(f"Objective target order mismatch: {split}")
                    for name in ("overall_quality", "geometry_quality", "texture_quality"):
                        raw[split][name] = objective[name].copy()
                    raw[split]["patch_quality"] = objective["patch_quality"].copy()
        self.dataset_provenance = dataset_provenance(
            dataset_manifest, raw, require_formal=require_formal)
        tile_by_key = {}
        if dataset_manifest is not None:
            manifest = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
            tile_by_key = {(str(row["asset_id"]), str(row["attack"]), str(row["level"])): row["sheet"]
                           for row in manifest["records"]}
        train_keys = set(zip(raw["train"]["asset_ids"].astype(str),
                             raw["train"]["attacks"].astype(str),
                             raw["train"]["levels"].astype(str)))
        train_tiles = sorted({value for key, value in tile_by_key.items() if key in train_keys})
        tile_index = {name: index for index, name in enumerate(train_tiles)}
        branch_stats = {}
        for branch in BRANCHES:
            value = raw["train"][branch].astype(np.float32)
            if normalization == "robust":
                center = np.median(value, axis=0)
                scale = (np.quantile(value, 0.75, axis=0) - np.quantile(value, 0.25, axis=0)) / 1.349
            else:
                center, scale = value.mean(0), value.std(0)
            branch_stats[branch] = (center, np.maximum(scale, 1e-5))
        valid_patch = raw["train"]["patches"][raw["train"]["patch_mask"]].astype(np.float32)
        patch_mean, patch_std = valid_patch.mean(0), np.maximum(valid_patch.std(0), 1e-5)
        self.data = {}
        for split, values in raw.items():
            item = {branch: torch.from_numpy(((values[branch] - branch_stats[branch][0]) /
                                              branch_stats[branch][1]).astype(np.float32)).to(device)
                    for branch in BRANCHES}
            patches = ((values["patches"] - patch_mean) / patch_std).astype(np.float32)
            patches[~values["patch_mask"]] = 0.0
            item.update({
                "patches": torch.from_numpy(patches).to(device),
                "patch_mask": torch.from_numpy(values["patch_mask"]).to(device),
                "attack": torch.from_numpy(values["attack_index"]).long().to(device),
                "severity": torch.from_numpy(values["severity"]).float().to(device),
                "overall": torch.from_numpy(values["overall_quality"]).float().to(device),
                "geometry": torch.from_numpy(values["geometry_quality"]).float().to(device),
                "texture_quality": torch.from_numpy(values["texture_quality"]).float().to(device),
                "patch_quality": torch.from_numpy(values.get(
                    "patch_quality", np.ones_like(values["patch_mask"], dtype=np.float32))).float().to(device),
                "asset_ids": values["asset_ids"].astype(str), "attacks": values["attacks"].astype(str),
                "levels": values["levels"].astype(str),
            })
            if tile_by_key:
                names = [tile_by_key[(str(a), str(b), str(c))] for a, b, c in zip(
                    values["asset_ids"], values["attacks"], values["levels"])]
                item["tiles"] = np.asarray(names)
                item["tile_index"] = torch.tensor(
                    [tile_index.get(name, -1) for name in names], dtype=torch.long, device=device)
            self.data[split] = item
        self.dims = [raw["train"][branch].shape[1] for branch in BRANCHES]
        self.statistics = {"branches": {branch: {"mean": branch_stats[branch][0].tolist(),
                                                   "std": branch_stats[branch][1].tolist()}
                                                for branch in BRANCHES},
                           "patch_mean": patch_mean.tolist(), "patch_std": patch_std.tolist(),
                           "normalization": normalization, "train_tiles": train_tiles}


class QualityHead(nn.Module):
    def __init__(self, dims, branch_indices, use_patches):
        super().__init__()
        self.branch_indices, self.use_patches = tuple(branch_indices), bool(use_patches)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dims[index]), nn.Linear(dims[index], 128), nn.GELU(),
                          nn.Linear(128, 128), nn.GELU()) for index in branch_indices])
        if use_patches:
            self.patch = nn.Sequential(nn.LayerNorm(58), nn.Linear(58, 128), nn.GELU(), nn.Linear(128, 128))
            self.patch_attention = nn.MultiheadAttention(128, 4, dropout=0.1, batch_first=True)
            self.patch_pool = nn.Linear(128, 1)
            self.patch_quality = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        count = len(branch_indices) + int(use_patches)
        self.embedding = nn.Parameter(torch.randn(1, count, 128) * 0.02)
        self.attention = nn.MultiheadAttention(128, 4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.shared = nn.Sequential(nn.Linear(128 * count, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1))
        self.attack = nn.Linear(256, len(ATTACKS))
        self.regression = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 4), nn.Sigmoid())

    def forward(self, branches, patches, patch_mask):
        tokens = [layer(branches[index]) for layer, index in zip(self.projections, self.branch_indices)]
        patch_weights = None
        patch_quality = None
        if self.use_patches:
            local = self.patch(patches)
            attended, _ = self.patch_attention(local, local, local, key_padding_mask=~patch_mask, need_weights=False)
            local = self.norm(local + attended)
            logits = self.patch_pool(local).squeeze(2).masked_fill(~patch_mask, -1e9)
            patch_weights = torch.softmax(logits, 1)
            patch_quality = self.patch_quality(local).squeeze(2)
            tokens.append(torch.sum(patch_weights[:, :, None] * local, 1))
        tokens = torch.stack(tokens, 1) + self.embedding
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        shared = self.shared(self.norm(tokens + attended).flatten(1))
        regression = self.regression(shared)
        return {"attack": self.attack(shared), "severity": regression[:, 0],
                "overall": regression[:, 1], "geometry": regression[:, 2],
                "texture": regression[:, 3], "patch_weights": patch_weights,
                "patch_quality": patch_quality}


def forward(model, data, index=None):
    if index is None:
        return model([data[b] for b in BRANCHES], data["patches"], data["patch_mask"])
    return model([data[b][index] for b in BRANCHES], data["patches"][index], data["patch_mask"][index])


def regression_metrics(estimate, target):
    estimate, target = np.asarray(estimate), np.asarray(target)
    return {"count": int(len(target)),
            "mae": float(np.mean(np.abs(estimate - target))),
            "plcc": correlation(estimate, target),
            "srcc": correlation(rankdata(estimate), rankdata(target))}


@torch.no_grad()
def evaluate(model, data):
    model.eval(); out = forward(model, data)
    truth = data["attack"].cpu().numpy(); predicted = out["attack"].argmax(1).cpu().numpy()
    macro, class_f1 = macro_f1(predicted, truth)
    result = {"count": len(truth), "accuracy": float(np.mean(predicted == truth)), "macro_f1": macro,
              "per_class_f1": {name: float(value) for name, value in zip(ATTACKS, class_f1)}}
    for key in ("severity", "overall", "geometry", "texture"):
        target_key = "texture_quality" if key == "texture" else key
        target = data[target_key].cpu().numpy(); estimate = out[key].cpu().numpy()
        result[key] = regression_metrics(estimate, target)
    overall_prediction = out["overall"].cpu().numpy()
    overall_truth = data["overall"].cpu().numpy()
    result["per_attack"] = {}
    for label, name in enumerate(ATTACKS):
        mask = truth == label
        result["per_attack"][name] = regression_metrics(
            overall_prediction[mask], overall_truth[mask])
    result["per_level_attacked_only"] = {}
    for level in ("light", "medium", "heavy"):
        mask = data["levels"] == level
        if np.any(mask):
            result["per_level_attacked_only"][level] = regression_metrics(
                overall_prediction[mask], overall_truth[mask])
    if "tiles" in data:
        result["per_tile"] = {}
        for tile in sorted(np.unique(data["tiles"]).tolist()):
            mask = data["tiles"] == tile
            result["per_tile"][tile] = regression_metrics(
                overall_prediction[mask], overall_truth[mask])
    if out["patch_quality"] is not None:
        local_mask = data["patch_mask"] & (
            (data["attack"] == 0) | (data["attack"] == 1) | (data["attack"] == 2) | (data["attack"] == 3)
        )[:, None]
        result["patch_quality_mae_geometry_and_clean"] = float(torch.mean(torch.abs(
            out["patch_quality"][local_mask] - data["patch_quality"][local_mask])).cpu())
    return result


def train_variant(name, store, args, device):
    seed_all(args.seed)
    indices, patches = VARIANTS[name]
    model = QualityHead(store.dims, indices, patches).to(device)
    initialization = None
    if args.init_checkpoint is not None:
        state = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        expected = (list(store.dims), tuple(indices), bool(patches))
        observed = (list(state["dims"]), tuple(state["branch_indices"]), bool(state["use_patches"]))
        if observed != expected:
            raise ValueError(f"Initialization checkpoint architecture mismatch: {observed} != {expected}")
        model.load_state_dict(state["model"])
        initialization = str(args.init_checkpoint)
    train, val = store.data["train"], store.data["val"]
    counts = torch.bincount(train["attack"], minlength=len(ATTACKS)).float()
    weights = counts.sum() / counts.clamp_min(1); weights /= weights.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_key, best_state, best_epoch, history = None, None, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.tile_balanced:
            if "tile_index" not in train:
                raise ValueError("--tile-balanced requires --dataset-manifest")
            groups = [torch.nonzero(train["tile_index"] == tile, as_tuple=False).flatten()
                      for tile in torch.unique(train["tile_index"]) if int(tile) >= 0]
            count = max(len(group) for group in groups)
            order = torch.cat([group[torch.randint(len(group), (count,), device=device)]
                               for group in groups])
            order = order[torch.randperm(len(order), device=device)]
        else:
            order = torch.randperm(len(train["attack"]), device=device)
        losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]; out = forward(model, train, index)
            per_sample = torch.zeros(len(index), device=device)
            if not args.quality_only:
                per_sample += 0.5 * F.cross_entropy(
                    out["attack"], train["attack"][index], weight=weights, reduction="none")
                per_sample += 0.5 * F.smooth_l1_loss(
                    out["severity"], train["severity"][index], reduction="none")
            per_sample += 3.0 * F.smooth_l1_loss(
                out["overall"], train["overall"][index], reduction="none")
            per_sample += 1.5 * F.smooth_l1_loss(
                out["geometry"], train["geometry"][index], reduction="none")
            per_sample += 1.5 * F.smooth_l1_loss(
                out["texture"], train["texture_quality"][index], reduction="none")
            if args.worst_tile:
                if "tile_index" not in train:
                    raise ValueError("--worst-tile requires --dataset-manifest")
                batch_tiles = train["tile_index"][index]
                group_losses = torch.stack([
                    per_sample[batch_tiles == tile].mean() for tile in torch.unique(batch_tiles)
                    if int(tile) >= 0
                ])
                loss = (torch.logsumexp(group_losses / args.worst_tile_temperature, 0)
                        - np.log(len(group_losses))) * args.worst_tile_temperature
            else:
                loss = per_sample.mean()
            target_delta = train["overall"][index][:, None] - train["overall"][index][None, :]
            valid_pairs = torch.abs(target_delta) > 0.05
            if valid_pairs.any():
                prediction_delta = out["overall"][:, None] - out["overall"][None, :]
                ranking = F.softplus(-5.0 * torch.sign(target_delta[valid_pairs])
                                     * prediction_delta[valid_pairs]).mean()
                loss = loss + 0.2 * ranking
            if out["patch_quality"] is not None:
                local_mask = train["patch_mask"][index] & (
                    (train["attack"][index] == 0) | (train["attack"][index] == 1)
                    | (train["attack"][index] == 2) | (train["attack"][index] == 3)
                )[:, None]
                loss = loss + 0.5 * F.smooth_l1_loss(
                    out["patch_quality"][local_mask], train["patch_quality"][index][local_mask])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        metrics = evaluate(model, val)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": metrics})
        key = (metrics["overall"]["srcc"], -metrics["overall"]["mae"],
               metrics["macro_f1"], -epoch)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"{name} epoch={epoch} loss={np.mean(losses):.4f} val_f1={metrics['macro_f1']:.4f} val_oq_mae={metrics['overall']['mae']:.4f} val_oq_srcc={metrics['overall']['srcc']:.4f}", flush=True)
    model.load_state_dict(best_state)
    evaluation_splits = ("val", "test", "blind") if args.evaluate_locked else ("val",)
    results = {split: evaluate(model, store.data[split]) for split in evaluation_splits}
    return model, {"variant": name, "best_epoch": best_epoch, "initialization": initialization,
                   "results": results, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objective-target-dir", type=Path)
    parser.add_argument("--dataset-manifest", type=Path,
                        help="Ordered dataset manifest used to audit every NPZ row")
    parser.add_argument("--require-formal", action="store_true",
                        help="Require the canonical 3493-record formal dataset")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Warm-start a matching QualityHead checkpoint")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quality-only", action="store_true",
                        help="Remove auxiliary degradation classification/strength losses")
    parser.add_argument("--normalization", choices=("mean_std", "robust"), default="mean_std")
    parser.add_argument("--tile-balanced", action="store_true")
    parser.add_argument("--worst-tile", action="store_true",
                        help="Use smooth worst-Train-tile loss within each batch")
    parser.add_argument("--worst-tile-temperature", type=float, default=0.1)
    parser.add_argument("--evaluate-locked", action="store_true",
                        help="Evaluate Test/Blind only after this run wins on Val")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    loaded_splits = ("train", "val", "test", "blind") if args.evaluate_locked else ("train", "val")
    store = Store(args.feature_dir, device, args.objective_target_dir,
                  args.dataset_manifest, args.require_formal, args.normalization,
                  splits=loaded_splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for variant in args.variants:
        model, summary = train_variant(variant, store, args, device)
        torch.save({"schema_version": 1, "seed": args.seed, "variant": variant,
                    "quality_only": args.quality_only,
                    "initialization": str(args.init_checkpoint) if args.init_checkpoint else None,
                    "training_strategy": {"normalization": args.normalization,
                                          "tile_balanced": args.tile_balanced,
                                          "worst_tile": args.worst_tile,
                                          "worst_tile_temperature": args.worst_tile_temperature},
                    "dataset_provenance": store.dataset_provenance,
                    "model": model.state_dict(), "dims": store.dims,
                    "branch_indices": VARIANTS[variant][0], "use_patches": VARIANTS[variant][1],
                    "statistics": store.statistics, "attacks": ATTACKS}, args.output_dir / f"{variant}.pt")
        summaries[variant] = summary
    best = max(summaries, key=lambda name: (
        summaries[name]["results"]["val"]["overall"]["srcc"],
        -summaries[name]["results"]["val"]["overall"]["mae"],
        summaries[name]["results"]["val"]["macro_f1"]))
    output = {"schema_version": 1, "status": "COMPLETE", "seed": args.seed,
              "protocol": {"single_seed": True, "selection": "validation only; test/blind locked",
                           "quality_only": args.quality_only,
                           "objective_supervision": args.objective_target_dir is not None,
                           "loaded_splits": list(loaded_splits),
                           "splits": {"train": 80, "val": 40, "test": 32, "blind": 32},
                           "dataset_provenance": store.dataset_provenance},
              "selected_variant": best, "variants": summaries}
    (args.output_dir / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": best, "results": {name: value["results"] for name, value in summaries.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
