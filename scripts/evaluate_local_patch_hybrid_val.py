#!/usr/bin/env python3
"""Evaluate a fixed local-head hybrid on Val without loading locked splits."""

import argparse, json
from pathlib import Path
import numpy as np
import torch

from train_local_patch_quality import GEOMETRY, TEXTURE, LocalPatchHead, load_split, metrics, tensorize


def load_model(checkpoint, device):
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    model=LocalPatchHead(state["use_atlas"],state["geometry_context"],state.get("cross_attention",False)).to(device).eval()
    model.load_state_dict(state["model"])
    return state,model


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--component-checkpoint",type=Path,required=True)
    parser.add_argument("--overall-checkpoint",type=Path,required=True); parser.add_argument("--feature-dir",type=Path,required=True)
    parser.add_argument("--target-dir",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda"); args=parser.parse_args(); device=torch.device(args.device)
    component_state,component=load_model(args.component_checkpoint,device); overall_state,overall=load_model(args.overall_checkpoint,device)
    if component_state["normalization"]!=overall_state["normalization"]: raise ValueError("Normalization differs")
    normalization={name:tuple(np.asarray(x,np.float32) for x in values) for name,values in component_state["normalization"].items()}
    data=tensorize(load_split(args.feature_dir,args.target_dir,"val"),normalization,device)
    with torch.no_grad(): component_out=component(data); overall_out=overall(data)
    prediction={"geometry":component_out["geometry"],"texture":component_out["texture"],"overall":overall_out["overall"]}
    patch=data["patch_mask"].cpu().numpy(); attacks=data["attacks"]
    masks={"geometry":patch&np.isin(attacks,list(GEOMETRY))[:,None],
           "texture":patch&data["texture_supervision"].cpu().numpy()&np.isin(attacks,list(TEXTURE))[:,None],"overall":patch}
    result={name:metrics(value.cpu().numpy(),data[f"{name}_target"].cpu().numpy(),masks[name]) for name,value in prediction.items()}
    report={"schema_version":1,"protocol":"fixed component/overall Val hybrid; Test/Blind not loaded","val":result,
            "component_checkpoint":str(args.component_checkpoint),"overall_checkpoint":str(args.overall_checkpoint)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report))


if __name__=="__main__": main()
