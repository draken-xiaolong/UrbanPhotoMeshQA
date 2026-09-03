#!/usr/bin/env python3
"""Align Patch Geometry/Texture/Overall targets to all 3493 formal records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

GEOMETRY={"geometry_hole","mesh_simplification_qem","geometry_noise_spike"}
TEXTURE={"texture_detail_loss","texture_region_missing","texture_misalignment"}


def load(root,prefix,quality):
    result={}
    for split in ("train","val","test","blind"):
        with np.load(root/f"{prefix}_{split}.npz") as values:
            keys=zip(values["asset_ids"].astype(str),values["attacks"].astype(str),values["levels"].astype(str))
            for index,key in enumerate(keys):
                result[key]={name:values[name][index].copy() for name in values.files
                             if name not in {"asset_ids","attacks","levels","metric_names"}}
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset-manifest",type=Path,required=True)
    parser.add_argument("--geometry-target-dir",type=Path,required=True); parser.add_argument("--texture-target-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args()
    payload=json.loads(args.dataset_manifest.read_text()); records=payload["records"]
    geometry=load(args.geometry_target_dir,"patch_geometry_targets","patch_geometry_quality")
    texture=load(args.texture_target_dir,"patch_texture_targets","patch_texture_quality")
    args.output_dir.mkdir(parents=True,exist_ok=True); counts={}
    digest=hashlib.sha256()
    for split in ("train","val","test","blind"):
        selected=[r for r in records if r["split"]==split]; counts[split]=len(selected)
        g=[]; t=[]; masks=[]; texture_masks=[]
        for row in selected:
            key=(row["asset_id"],row["attack"],row["level"]); clean=(row["asset_id"],"clean","clean")
            gv=geometry[key if row["attack"] in GEOMETRY|{"clean"} else clean]
            tv=texture[key if row["attack"] in TEXTURE|{"clean"} else clean]
            g.append(gv["patch_geometry_quality"] if row["attack"] in GEOMETRY else np.ones(16,np.float32))
            t.append(tv["patch_texture_quality"] if row["attack"] in TEXTURE else np.ones(16,np.float32))
            masks.append(gv["patch_mask"]); texture_masks.append(tv["texture_supervision_mask"])
            digest.update("|".join(key).encode()); digest.update(b"\n")
        geometry_values=np.stack(g).astype(np.float32); texture_values=np.stack(t).astype(np.float32)
        patch_mask=np.stack(masks).astype(bool); texture_mask=np.stack(texture_masks).astype(bool)
        overall=np.minimum(geometry_values,texture_values)
        np.savez_compressed(args.output_dir/f"local_patch_targets_{split}.npz",
            asset_ids=np.asarray([r["asset_id"] for r in selected]),attacks=np.asarray([r["attack"] for r in selected]),
            levels=np.asarray([r["level"] for r in selected]),patch_geometry_quality=geometry_values,
            patch_texture_quality=texture_values,patch_overall_quality=overall,patch_mask=patch_mask,
            texture_supervision_mask=texture_mask)
    audit={"status":"PASSED","counts":counts,"records":sum(counts.values()),
           "ordered_sample_key_sha256":digest.hexdigest(),"geometry_records":len(geometry),"texture_records":len(texture),
           "formula":"Patch Overall Quality = min(Patch Geometry Quality, Patch Texture Quality)",
           "protocol":"Clean/Attacked pairing only for offline targets; inference remains single-model no-reference"}
    (args.output_dir/"metadata.json").write_text(json.dumps(audit,indent=2)+"\n"); print(json.dumps(audit))


if __name__=="__main__": main()
