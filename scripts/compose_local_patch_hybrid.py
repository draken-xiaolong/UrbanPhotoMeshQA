#!/usr/bin/env python3
"""Compose the Val-best geometry and texture branches into one local checkpoint."""

from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch

from train_local_patch_quality import (GEOMETRY,TEXTURE,LocalPatchHead,load_split,metrics,tensorize)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--geometry-checkpoint",type=Path,required=True)
    parser.add_argument("--texture-checkpoint",type=Path,required=True); parser.add_argument("--feature-dir",type=Path,required=True)
    parser.add_argument("--target-dir",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda"); args=parser.parse_args(); device=torch.device(args.device)
    geometry=torch.load(args.geometry_checkpoint,map_location=device,weights_only=False)
    texture=torch.load(args.texture_checkpoint,map_location=device,weights_only=False)
    if geometry["normalization"]!=texture["normalization"]: raise ValueError("Checkpoint normalization differs")
    state=geometry["model"].copy(); prefixes=("view.","atlas.","texture.","texture_quality.")
    for name,value in texture["model"].items():
        if name.startswith(prefixes): state[name]=value
    model=LocalPatchHead(use_atlas=True,geometry_context=True).to(device).eval(); model.load_state_dict(state)
    normalization={name:tuple(np.asarray(x,np.float32) for x in values)
                   for name,values in geometry["normalization"].items()}
    raw=load_split(args.feature_dir,args.target_dir,"val"); data=tensorize(raw,normalization,device)
    with torch.no_grad(): out=model(data)
    patch=data["patch_mask"].cpu().numpy(); attacks=data["attacks"]
    masks={"geometry":patch&np.isin(attacks,list(GEOMETRY))[:,None],
           "texture":patch&data["texture_supervision"].cpu().numpy()&np.isin(attacks,list(TEXTURE))[:,None],
           "overall":patch}
    result={name:metrics(value.cpu().numpy(),data[f"{name}_target"].cpu().numpy(),masks[name])
            for name,value in out.items()}
    target_metadata=geometry.get("target_metadata",{}); feature_metadata=geometry.get("feature_metadata",{})
    payload={"model":state,"use_atlas":True,"geometry_context":True,"normalization":geometry["normalization"],
             "seed":2026,"val":result,"composition":{"geometry_checkpoint":str(args.geometry_checkpoint),
             "texture_checkpoint":str(args.texture_checkpoint),
             "geometry_checkpoint_sha256":hashlib.sha256(args.geometry_checkpoint.read_bytes()).hexdigest(),
             "texture_checkpoint_sha256":hashlib.sha256(args.texture_checkpoint.read_bytes()).hexdigest()},
             "formal_ordered_sample_key_sha256":target_metadata.get("ordered_sample_key_sha256"),
             "formal_counts":target_metadata.get("counts"),"feature_metadata":feature_metadata,
             "target_metadata":target_metadata,"test_blind_loaded":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,args.output)
    args.output.with_suffix(".json").write_text(json.dumps({k:v for k,v in payload.items() if k!="model"},indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__": main()
