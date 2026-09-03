#!/usr/bin/env python3
"""Extract topology and visible local-texture features from individual glTFs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.local_features import extract_local_features
from urbanphotomeshqa.patches import topological_patch_layout
from urbanphotomeshqa.texture import STANDARD_DIRECTIONS
from urbanphotomeshqa.texture_features import SpatialImageEncoder


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--splits",nargs="+",default=("train","val"))
    parser.add_argument("--render-size",type=int,default=224); parser.add_argument("--shard-index",type=int,default=0)
    parser.add_argument("--shard-count",type=int,default=1); parser.add_argument("--device",default="cuda")
    parser.add_argument("--asset-ids",nargs="*"); parser.add_argument("--limit",type=int)
    parser.add_argument("--allow-locked-evaluation",action="store_true")
    args=parser.parse_args(); device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    if not set(args.splits) <= {"train", "val"} and not args.allow_locked_evaluation:
        raise ValueError("Candidate local feature extraction is restricted to Train/Val")
    manifest_bytes=args.manifest.read_bytes(); requested=set(args.asset_ids or [])
    records=[r for r in json.loads(manifest_bytes)["records"]
             if r["split"] in set(args.splits) and (not requested or r["asset_id"] in requested)]
    assets=sorted({row["asset_id"] for row in records})[args.shard_index::args.shard_count]
    records=[row for row in records if row["asset_id"] in set(assets)]
    encoder=SpatialImageEncoder(device); rows=[]; clean_layouts={}
    if args.limit is not None: records=records[:args.limit]
    for index,row in enumerate(records):
        mesh=GltfReader(row["gltf_path"]).load_mesh(include_texture=True)
        layout=None
        if row["attack"]=="clean":
            layout=topological_patch_layout(mesh,16); clean_layouts[row["asset_id"]]=layout
        elif row["attack"].startswith("texture_"):
            layout=clean_layouts[row["asset_id"]]
        rows.append({"row":row,**extract_local_features(mesh,encoder,args.render_size,layout)})
        print(f"feature {index+1}/{len(records)} {row['asset_id']} {row['attack']} {row['level']}",flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    arrays=("patch_descriptors","patch_mask","patch_area","patch_center","patch_view_tokens","patch_view_mask","patch_view_stats",
            "patch_atlas_tokens","patch_atlas_mask","patch_atlas_stats")
    for split in args.splits:
        selected=[r for r in rows if r["row"]["split"]==split]
        payload={"asset_ids":np.asarray([r["row"]["asset_id"] for r in selected]),
                 "attacks":np.asarray([r["row"]["attack"] for r in selected]),
                 "levels":np.asarray([r["row"]["level"] for r in selected])}
        for name in arrays: payload[name]=np.stack([r[name] for r in selected]) if selected else np.empty((0,))
        np.savez_compressed(args.output_dir/f"local_patch_features_{split}.npz",**payload)
    repository=Path(__file__).resolve().parents[1]
    source_files=[Path(__file__).resolve(),repository/"src/urbanphotomeshqa/texture_features.py",
                  repository/"src/urbanphotomeshqa/texture.py",repository/"src/urbanphotomeshqa/patches.py",
                  repository/"src/urbanphotomeshqa/gltf.py",repository/"src/urbanphotomeshqa/local_features.py"]
    source_sha={str(path.relative_to(repository)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_files}
    sample_digest=hashlib.sha256("".join("|".join((r["row"]["asset_id"],r["row"]["attack"],r["row"]["level"]))+"\n"
                                          for r in rows).encode()).hexdigest()
    metadata={"schema_version":1,"splits":args.splits,"records":len(rows),"render_size":args.render_size,
              "patches":16,"views":len(STANDARD_DIRECTIONS),"texture_encoder":"MobileNet_V3_Small frozen",
              "shard_index":args.shard_index,"shard_count":args.shard_count,
              "test_blind_loaded":bool(set(args.splits)&{"test","blind"}),
              "locked_evaluation_authorized":args.allow_locked_evaluation,
              "sharding":"asset-level; clean topology reused only for file-native texture attacks",
              "manifest_sha256":hashlib.sha256(manifest_bytes).hexdigest(),
              "ordered_sample_key_sha256":sample_digest,"source_sha256":source_sha,
              "implementation_sha256":hashlib.sha256("".join(source_sha.values()).encode()).hexdigest()}
    (args.output_dir/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n"); print(json.dumps({"status":"COMPLETE",**metadata}))


if __name__=="__main__": main()
