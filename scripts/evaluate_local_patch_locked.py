#!/usr/bin/env python3
"""One-time Test/Blind evaluation of a frozen local Patch checkpoint."""

from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch

from train_local_patch_quality import LocalPatchHead,evaluate,load_split,tensorize


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--feature-dir",type=Path,required=True); parser.add_argument("--target-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--device",default="cuda")
    args=parser.parse_args(); device=torch.device(args.device); state=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=LocalPatchHead(state["use_atlas"],state["geometry_context"]).to(device).eval(); model.load_state_dict(state["model"])
    normalization={name:tuple(np.asarray(x,np.float32) for x in values) for name,values in state["normalization"].items()}
    results={}
    for split in ("test","blind"):
        raw=load_split(args.feature_dir,args.target_dir,split); results[split]=evaluate(model,tensorize(raw,normalization,device))
    report={"schema_version":1,"checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "protocol":"one-time locked evaluation after Val selection; no further model changes", "results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report))


if __name__=="__main__": main()
