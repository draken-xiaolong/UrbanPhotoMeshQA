#!/usr/bin/env python3
"""Compose and audit Patch geometry-target shards in formal order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

ATTACKS={"geometry_hole","mesh_simplification_qem","geometry_noise_spike"}
ARRAYS=("patch_geometry_quality","patch_geometry_quality_raw","patch_metrics",
        "directional_sample_count","objective_noop","patch_mask")
LEVEL={"light":0,"medium":1,"heavy":2}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset-manifest",type=Path,required=True)
    parser.add_argument("--shard-dirs",type=Path,nargs="+",required=True); parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(); manifest=json.loads(args.dataset_manifest.read_text())["records"]
    lookup={}; metric_names=None; metadata=[]
    for root in args.shard_dirs:
        metadata.append(json.loads((root/"metadata.json").read_text()))
        for split in ("train","val","test","blind"):
            with np.load(root/f"patch_geometry_targets_{split}.npz") as values:
                names=values["metric_names"].astype(str).tolist()
                if metric_names is None: metric_names=names
                if metric_names!=names: raise ValueError(f"Metric mismatch: {root}/{split}")
                keys=zip(values["asset_ids"].astype(str),values["attacks"].astype(str),values["levels"].astype(str))
                for index,key in enumerate(keys):
                    if key in lookup: raise ValueError(f"Duplicate key: {key}")
                    lookup[key]={name:values[name][index].copy() for name in ARRAYS}
    args.output_dir.mkdir(parents=True,exist_ok=True); ordered=[]; counts={}
    for split in ("train","val","test","blind"):
        records=[r for r in manifest if r["split"]==split and r["attack"] in ATTACKS|{"clean"}]
        keys=[(r["asset_id"],r["attack"],r["level"]) for r in records]; missing=[k for k in keys if k not in lookup]
        if missing: raise ValueError(f"Missing {len(missing)} {split} keys; first={missing[0]}")
        counts[split]=len(keys); ordered.extend(keys)
        payload={"asset_ids":np.asarray([k[0] for k in keys]),"attacks":np.asarray([k[1] for k in keys]),
                 "levels":np.asarray([k[2] for k in keys]),"metric_names":np.asarray(metric_names)}
        for name in ARRAYS: payload[name]=np.stack([lookup[k][name] for k in keys])
        np.savez_compressed(args.output_dir/f"patch_geometry_targets_{split}.npz",**payload)
    if set(ordered)!=set(lookup): raise ValueError("Shard output contains non-formal keys")
    clean_ok=all(np.allclose(lookup[k]["patch_geometry_quality"],1) for k in ordered if k[1]=="clean")
    finite=all(np.isfinite(lookup[k]["patch_geometry_quality"]).all() for k in ordered)
    audit={"status":"PASSED" if clean_ok and finite else "FAILED","counts":counts,
           "records":len(ordered),"assets":len({k[0] for k in ordered}),"clean_quality_one":clean_ok,
           "finite_quality":finite,
           "local_monotonic_envelope": "not applied because independently partitioned attacked meshes do not preserve Patch-index correspondence across severity"}
    report={**metadata[0],"scope":"formal","assets":audit["assets"],"records":len(ordered),
            "composed_from_shards":len(args.shard_dirs),"shard_metadata":metadata,"audit":audit}
    (args.output_dir/"metadata.json").write_text(json.dumps(report,indent=2)+"\n"); (args.output_dir/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit))


if __name__=="__main__": main()
