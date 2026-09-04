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
    parser.add_argument("--spatial-texture-dir",type=Path); parser.add_argument("--spatial-hierarchical",type=Path)
    parser.add_argument("--augmented-hierarchical",type=Path)
    parser.add_argument("--batch-size",type=int,default=8); parser.add_argument("--device",default="cuda")
    args=parser.parse_args(); device=torch.device(args.device); train=load_split("train",args); val=load_split("val",args)
    for name in ("point","morph","texture"):
        mean=train[name].mean(0); std=np.maximum(train[name].std(0),1e-5)
        train[name]=(train[name]-mean)/std; val[name]=(val[name]-mean)/std
    mean=train["texture_views"].reshape(-1,train["texture_views"].shape[-1]).mean(0)
    std=np.maximum(train["texture_views"].reshape(-1,train["texture_views"].shape[-1]).std(0),1e-5)
    train["texture_views"]=(train["texture_views"]-mean)/std; val["texture_views"]=(val["texture_views"]-mean)/std
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
        report["ensemble3_equal_diagnostics"]={
            "by_attack":subgroup_metrics(ensemble3,val,np.asarray([key[1] for key in val["keys"]])),
            "by_level":subgroup_metrics(ensemble3,val,val["levels"]),
            "by_tile":subgroup_metrics(ensemble3,val,val["tiles"])}
        train_predictions=[predict(model_item,train,args.batch_size,device) for model_item in (*models,model)]
        train_ensemble3={name:sum(prediction[name] for prediction in train_predictions)/3 for name in train_predictions[0]}
        report["train"]={"ensemble3_equal":metrics_from_output(train_ensemble3,train),
                         "ensemble3_equal_diagnostics":{
                             "by_attack":subgroup_metrics(train_ensemble3,train,np.asarray([key[1] for key in train["keys"]])),
                             "by_level":subgroup_metrics(train_ensemble3,train,train["levels"]),
                             "by_tile":subgroup_metrics(train_ensemble3,train,train["tiles"])}}
        augmented_prediction = None
        if args.augmented_hierarchical:
            augmented_model=HierarchicalQualityGAT(val["point"].shape[1],val["morph"].shape[1],val["texture"].shape[1]).to(device)
            augmented_model.load_state_dict(torch.load(args.augmented_hierarchical,map_location=device,weights_only=False)["model"])
            augmented_prediction=predict(augmented_model,val,args.batch_size,device)
            deploy4={name:(ensemble3[name]*3+augmented_prediction[name])/4 for name in ensemble3}
            report["augmented_hierarchical"]=metrics_from_output(augmented_prediction,val)
            report["deployable_ensemble4_equal"]=metrics_from_output(deploy4,val)
            report["deployable_ensemble4_equal_diagnostics"]={
                "by_attack":subgroup_metrics(deploy4,val,np.asarray([key[1] for key in val["keys"]])),
                "by_level":subgroup_metrics(deploy4,val,val["levels"]),
                "by_tile":subgroup_metrics(deploy4,val,val["tiles"])}
        if args.spatial_texture_dir and args.spatial_hierarchical:
            original_texture_dir=args.texture_dir; args.texture_dir=args.spatial_texture_dir
            spatial_train=load_split("train",args); spatial_val=load_split("val",args); args.texture_dir=original_texture_dir
            for name in ("point","morph","texture"):
                spatial_mean=spatial_train[name].mean(0); spatial_std=np.maximum(spatial_train[name].std(0),1e-5)
                spatial_val[name]=(spatial_val[name]-spatial_mean)/spatial_std
            spatial_mean=spatial_train["texture_views"].reshape(-1,spatial_train["texture_views"].shape[-1]).mean(0)
            spatial_std=np.maximum(spatial_train["texture_views"].reshape(-1,spatial_train["texture_views"].shape[-1]).std(0),1e-5)
            spatial_val["texture_views"]=(spatial_val["texture_views"]-spatial_mean)/spatial_std
            spatial_model=HierarchicalQualityGAT(spatial_val["point"].shape[1],spatial_val["morph"].shape[1],spatial_val["texture"].shape[1]).to(device)
            spatial_model.load_state_dict(torch.load(args.spatial_hierarchical,map_location=device,weights_only=False)["model"])
            spatial_prediction=predict(spatial_model,spatial_val,args.batch_size,device)
            ensemble4={name:(ensemble3[name]*3+spatial_prediction[name])/4 for name in ensemble3}
            report["spatial_hierarchical"]=metrics_from_output(spatial_prediction,spatial_val)
            report["ensemble4_equal"]=metrics_from_output(ensemble4,val)
            report["ensemble4_equal_diagnostics"]={
                "by_attack":subgroup_metrics(ensemble4,val,np.asarray([key[1] for key in val["keys"]])),
                "by_level":subgroup_metrics(ensemble4,val,val["levels"]),
                "by_tile":subgroup_metrics(ensemble4,val,val["tiles"])}
            if augmented_prediction is not None:
                ensemble5={name:(ensemble4[name]*4+augmented_prediction[name])/5 for name in ensemble4}
                report["ensemble5_equal"]=metrics_from_output(ensemble5,val)
                report["ensemble5_equal_diagnostics"]={
                    "by_attack":subgroup_metrics(ensemble5,val,np.asarray([key[1] for key in val["keys"]])),
                    "by_level":subgroup_metrics(ensemble5,val,val["levels"]),
                    "by_tile":subgroup_metrics(ensemble5,val,val["tiles"])}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__": main()
