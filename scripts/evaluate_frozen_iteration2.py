#!/usr/bin/env python3
"""One-shot evaluation of a pre-frozen Iteration-2 ensemble on locked splits."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from evaluate_gat_ensemble_val import metrics_from_output, predict, subgroup_metrics
from train_gatv2_mesh_quality import HierarchicalQualityGAT, QualityGAT


def keys(values):
    return list(zip(values["asset_ids"].astype(str), values["attacks"].astype(str), values["levels"].astype(str)))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--release-dir",type=Path,required=True)
    p.add_argument("--feature-dir",type=Path,required=True); p.add_argument("--texture-dir",type=Path,required=True)
    p.add_argument("--target-dir",type=Path,required=True); p.add_argument("--cache-audit",type=Path,required=True)
    p.add_argument("--splits",nargs="+",default=("test","blind")); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--device",default="cuda")
    args=p.parse_args(); device=torch.device(args.device)
    release=json.loads((args.release_dir/"release_manifest.json").read_text()); norm=release["normalization"]["base"]
    audit=json.loads(args.cache_audit.read_text())["records"]
    paths={(r["asset_id"],r["attack"],r["level"]):r["cache_path"] for r in audit}
    tiles={(r["asset_id"],r["attack"],r["level"]):r["sheet"] for r in audit}
    report={"schema_version":1,"release_status_before_evaluation":release["status"],
            "protocol":"one-shot frozen Test/Blind evaluation; no selection or parameter changes"}
    for split in args.splits:
        with np.load(args.feature_dir/f"features_{split}.npz") as z: base={n:z[n].copy() for n in z.files}
        with np.load(args.texture_dir/f"strong_texture_{split}.npz") as z: texture={n:z[n].copy() for n in z.files}
        with np.load(args.target_dir/f"objective_targets_{split}.npz") as z: target={n:z[n].copy() for n in z.files}
        sample_keys=keys(base)
        if sample_keys != keys(texture) or sample_keys != keys(target): raise ValueError(f"Order mismatch: {split}")
        graphs=[]
        for i,key in enumerate(sample_keys):
            with np.load(paths[key]) as z: graphs.append({n:z[n].copy() for n in ("face_features","neighbors","topology")})
            if (i+1)%500==0: print(f"load {split} {i+1}/{len(sample_keys)}",flush=True)
        data={"keys":sample_keys,"graphs":graphs,"faces":np.asarray([len(g["face_features"]) for g in graphs]),
              "tiles":np.asarray([tiles[k] for k in sample_keys]),"levels":base["levels"].astype(str),
              "point":base["point_identity"].astype(np.float32),"morph":base["morphology"].astype(np.float32),
              "texture":texture["texture"].astype(np.float32),"texture_views":texture["texture_views"].astype(np.float32),
              "attack":base["attack_index"].astype(np.int64),"severity":base["severity"].astype(np.float32),
              "overall":target["overall_quality"].astype(np.float32),"geometry":target["geometry_quality"].astype(np.float32),
              "texture_target":target["texture_quality"].astype(np.float32),"ordinal":target["ordinal_grade"].astype(np.int64)}
        for name in ("point","morph","texture","texture_views"):
            data[name]=(data[name]-np.asarray(norm[name]["mean"],np.float32))/np.asarray(norm[name]["std"],np.float32)
        models=[]
        for i,name in enumerate(release["model_order"]):
            cls=QualityGAT if i==0 else HierarchicalQualityGAT
            model=cls(data["point"].shape[1],data["morph"].shape[1],data["texture"].shape[1]).to(device)
            model.load_state_dict(torch.load(args.release_dir/"models"/name,map_location=device,weights_only=False)["model"]); models.append(model)
        outputs=[predict(model,data,args.batch_size,device) for model in models]
        ensemble={name:sum(value[name] for value in outputs)/len(outputs) for name in outputs[0]}
        report[split]={"count":len(sample_keys),"metrics":metrics_from_output(ensemble,data),
                       "diagnostics":{"by_attack":subgroup_metrics(ensemble,data,np.asarray([k[1] for k in sample_keys])),
                                      "by_level":subgroup_metrics(ensemble,data,data["levels"]),
                                      "by_tile":subgroup_metrics(ensemble,data,data["tiles"])}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({s:report[s]["metrics"] for s in args.splits}))


if __name__=="__main__": main()
