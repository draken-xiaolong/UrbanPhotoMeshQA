#!/usr/bin/env python3
"""Audit live single-glTF local inference against stored feature cache."""

from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch

from train_local_patch_quality import LocalPatchHead,tensorize_features
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.local_features import extract_local_features
from urbanphotomeshqa.texture_features import SpatialImageEncoder

ARRAYS=("patch_descriptors","patch_mask","patch_area","patch_center","patch_view_tokens",
        "patch_view_mask","patch_view_stats","patch_atlas_tokens","patch_atlas_mask","patch_atlas_stats")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--feature-dir",type=Path,required=True); parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--asset-ids",nargs="+",required=True)
    parser.add_argument("--device",default="cuda"); args=parser.parse_args(); device=torch.device(args.device)
    rows=[r for r in json.loads(args.manifest.read_text())["records"] if r["asset_id"] in set(args.asset_ids)]
    if not rows or any(r["split"] not in {"train","val"} for r in rows): raise ValueError("Audit is restricted to Train/Val assets")
    wanted={(r["asset_id"],r["attack"],r["level"]) for r in rows}; cache={}
    for split in sorted({r["split"] for r in rows}):
        with np.load(args.feature_dir/f"local_patch_features_{split}.npz") as values:
            keys=list(zip(values["asset_ids"].astype(str),values["attacks"].astype(str),values["levels"].astype(str)))
            indices=[index for index,key in enumerate(keys) if key in wanted]
            loaded={name:values[name][indices].copy() for name in ARRAYS}
            for position,index in enumerate(indices):
                cache[keys[index]]={name:loaded[name][position] for name in ARRAYS}
    state=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=LocalPatchHead(state["use_atlas"],state["geometry_context"],state.get("cross_attention",False)).to(device).eval(); model.load_state_dict(state["model"])
    normalization={name:tuple(np.asarray(x,np.float32) for x in value) for name,value in state["normalization"].items()}
    encoder=SpatialImageEncoder(device); report=[]
    with torch.no_grad():
        for index,row in enumerate(rows):
            key=(row["asset_id"],row["attack"],row["level"]); stored=cache[key]
            live=extract_local_features(GltfReader(row["gltf_path"]).load_mesh(include_texture=True),encoder,224)
            differences={name:(0. if np.array_equal(stored[name],live[name]) else
                float(np.max(np.abs(stored[name].astype(np.float64)-live[name].astype(np.float64))))) for name in ARRAYS}
            batches=[]
            for values in (stored,live): batches.append(tensorize_features({name:value[None] for name,value in values.items()},normalization,device))
            predictions=[model(values) for values in batches]
            prediction_difference={name:float(torch.max(torch.abs(predictions[0][name]-predictions[1][name])).cpu())
                                   for name in predictions[0]}
            report.append({"key":key,"array_max_absolute_difference":differences,
                           "prediction_max_absolute_difference":prediction_difference})
            print(f"audit {index+1}/{len(rows)} {key}",flush=True)
    maximum=max(value for row in report for value in row["prediction_max_absolute_difference"].values())
    result={"status":"PASSED" if maximum==0 else "FAILED","records":len(report),
            "maximum_prediction_absolute_difference":maximum,"details":report,"test_blind_loaded":False}
    result["checkpoint_sha256"]=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="details"}))


if __name__=="__main__": main()
