#!/usr/bin/env python3
"""Compose Train/Val local Patch feature shards in formal order."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

ARRAYS=("patch_descriptors","patch_mask","patch_area","patch_center","patch_view_tokens",
        "patch_view_mask","patch_view_stats","patch_atlas_tokens","patch_atlas_mask","patch_atlas_stats")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--shard-dirs",type=Path,nargs="+",required=True); parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--splits",nargs="+",default=("train","val"))
    args=parser.parse_args(); manifest=json.loads(args.manifest.read_text())["records"]; lookup={}; metadata=[]
    for root in args.shard_dirs:
        metadata.append(json.loads((root/"metadata.json").read_text()))
        for split in args.splits:
            with np.load(root/f"local_patch_features_{split}.npz") as values:
                keys=zip(values["asset_ids"].astype(str),values["attacks"].astype(str),values["levels"].astype(str))
                for index,key in enumerate(keys):
                    if key in lookup: raise ValueError(f"Duplicate key: {key}")
                    lookup[key]={name:values[name][index].copy() for name in ARRAYS}
    args.output_dir.mkdir(parents=True,exist_ok=True); counts={}; finite=True
    for split in args.splits:
        records=[r for r in manifest if r["split"]==split]; keys=[(r["asset_id"],r["attack"],r["level"]) for r in records]
        missing=[k for k in keys if k not in lookup]
        if missing: raise ValueError(f"Missing {len(missing)} {split} keys; first={missing[0]}")
        counts[split]=len(keys); payload={"asset_ids":np.asarray([k[0] for k in keys]),
            "attacks":np.asarray([k[1] for k in keys]),"levels":np.asarray([k[2] for k in keys])}
        for name in ARRAYS: payload[name]=np.stack([lookup[k][name] for k in keys])
        finite &= all(np.isfinite(payload[name]).all() for name in
                      ("patch_descriptors","patch_area","patch_center","patch_view_tokens","patch_view_stats",
                       "patch_atlas_tokens","patch_atlas_stats"))
        # Frozen neural tokens are high-entropy and barely compress; an
        # uncompressed NPZ is much faster to compose and load for training.
        np.savez(args.output_dir/f"local_patch_features_{split}.npz",**payload)
    expected={name:sum(row["split"]==name for row in manifest) for name in args.splits}
    audit={"status":"PASSED" if finite and counts==expected else "FAILED",
           "counts":counts,"records":sum(counts.values()),"finite":bool(finite),
           "test_blind_loaded":bool(set(args.splits)&{"test","blind"}),
           "shards":len(args.shard_dirs),"shard_metadata":metadata}
    (args.output_dir/"metadata.json").write_text(json.dumps(audit,indent=2)+"\n"); print(json.dumps(audit))


if __name__=="__main__": main()
