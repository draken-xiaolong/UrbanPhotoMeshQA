#!/usr/bin/env python3
"""Evaluate a fixed equal-weight Val ensemble without touching locked splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_gatv2_mesh_quality import (HierarchicalQualityGAT, QualityGAT, load_split,
                                      metrics_from_output, predict)


def subgroup_metrics(output, data, values):
    result={}
    for value in np.unique(values):
        mask=values==value
        if mask.sum()<3: continue
        subset={name:(array[mask] if isinstance(array,np.ndarray) and len(array)==len(mask) else array)
                for name,array in data.items()}
        pred={name:array[mask] for name,array in output.items()}
        result[str(value)]={"count":int(mask.sum()),**metrics_from_output(pred,subset)}
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--feature-dir",type=Path,required=True)
    parser.add_argument("--texture-dir",type=Path,required=True); parser.add_argument("--target-dir",type=Path,required=True)
    parser.add_argument("--cache-audit",type=Path,required=True); parser.add_argument("--gat",type=Path,required=True)
    parser.add_argument("--hierarchical",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--balanced-hierarchical",type=Path)
    parser.add_argument("--batch-size",type=int,default=8); parser.add_argument("--device",default="cuda")
    args=parser.parse_args(); device=torch.device(args.device); train=load_split("train",args); val=load_split("val",args)
    for name in ("point","morph","texture"):
        mean=train[name].mean(0); std=np.maximum(train[name].std(0),1e-5); val[name]=(val[name]-mean)/std
    mean=train["texture_views"].reshape(-1,train["texture_views"].shape[-1]).mean(0)
    std=np.maximum(train["texture_views"].reshape(-1,train["texture_views"].shape[-1]).std(0),1e-5)
    val["texture_views"]=(val["texture_views"]-mean)/std
    models=[]
    for architecture,path in (("gatv2",args.gat),("hierarchical",args.hierarchical)):
        cls=QualityGAT if architecture=="gatv2" else HierarchicalQualityGAT
        model=cls(val["point"].shape[1],val["morph"].shape[1],val["texture"].shape[1]).to(device)
        model.load_state_dict(torch.load(path,map_location=device,weights_only=False)["model"]); models.append(model)
    predictions=[predict(model,val,args.batch_size,device) for model in models]
    ensemble={name:0.5*predictions[0][name]+0.5*predictions[1][name] for name in predictions[0]}
    diagnostics={"by_attack":subgroup_metrics(ensemble,val,np.asarray([key[1] for key in val["keys"]])),
                 "by_level":subgroup_metrics(ensemble,val,val["levels"]),
                 "by_tile":subgroup_metrics(ensemble,val,val["tiles"])}
    report={"schema_version":1,"protocol":"fixed 0.5/0.5 Val ensemble; Test/Blind not loaded",
            "gat":metrics_from_output(predictions[0],val),"hierarchical":metrics_from_output(predictions[1],val),
            "ensemble":metrics_from_output(ensemble,val),"ensemble_diagnostics":diagnostics}
    if args.balanced_hierarchical:
        model=HierarchicalQualityGAT(val["point"].shape[1],val["morph"].shape[1],val["texture"].shape[1]).to(device)
        model.load_state_dict(torch.load(args.balanced_hierarchical,map_location=device,weights_only=False)["model"])
        balanced=predict(model,val,args.batch_size,device)
        ensemble3={name:sum(prediction[name] for prediction in (*predictions,balanced))/3 for name in predictions[0]}
        report["balanced_hierarchical"]=metrics_from_output(balanced,val)
        report["ensemble3_equal"]=metrics_from_output(ensemble3,val)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__": main()
