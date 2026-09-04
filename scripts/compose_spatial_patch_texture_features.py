#!/usr/bin/env python3
"""Compose 16 spatial UV-Patch texture tokens for global quality training."""

import argparse, json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--local-feature-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--splits",nargs="+",default=("train","val"))
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); counts={}
    for split in args.splits:
        with np.load(args.local_feature_dir/f"local_patch_features_{split}.npz") as source:
            patch=source["patch_mask"].astype(bool); view_mask=source["patch_view_mask"].astype(bool)&patch[:,:,None]
            view_count=np.maximum(view_mask.sum(2,keepdims=True),1)
            view_embedding=(source["patch_view_tokens"]*view_mask[:,:,:,None]).sum(2)/view_count
            view_stats=(source["patch_view_stats"]*view_mask[:,:,:,None]).sum(2)/view_count
            atlas_mask=source["patch_atlas_mask"].astype(bool)&patch
            atlas_embedding=source["patch_atlas_tokens"]*atlas_mask[:,:,None]
            atlas_stats=source["patch_atlas_stats"]*atlas_mask[:,:,None]
            tokens=np.concatenate([view_embedding,view_stats,atlas_embedding,atlas_stats],axis=2).astype(np.float32)
            pooled=(tokens*patch[:,:,None]).sum(1)/np.maximum(patch.sum(1,keepdims=True),1)
            payload={"asset_ids":source["asset_ids"].copy(),"attacks":source["attacks"].copy(),
                     "levels":source["levels"].copy(),"texture":pooled.astype(np.float32),
                     "texture_views":tokens,"patch_mask":patch}
        np.savez(args.output_dir/f"strong_texture_{split}.npz",**payload); counts[split]=len(pooled)
    metadata={"schema_version":1,"counts":counts,"tokens":16,"token_semantics":"visible-view + UV-atlas per Mesh Patch",
              "test_blind_loaded":bool(set(args.splits)&{"test","blind"})}
    (args.output_dir/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n"); print(json.dumps(metadata))


if __name__=="__main__": main()
