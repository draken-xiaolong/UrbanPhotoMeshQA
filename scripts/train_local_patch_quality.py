#!/usr/bin/env python3
"""Train Val-selected no-reference Geometry/Texture/Overall Patch heads."""

from __future__ import annotations
import argparse,copy,json,random
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn as nn
import torch.nn.functional as F

GEOMETRY={"clean","geometry_hole","mesh_simplification_qem","geometry_noise_spike"}
TEXTURE={"clean","texture_detail_loss","texture_region_missing","texture_misalignment"}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def corr(a,b):
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12: return 0.
    return float(np.corrcoef(a,b)[0,1])


def metrics(pred,target,mask):
    p=pred[mask]; t=target[mask]
    return {"count":int(len(t)),"mae":float(np.mean(np.abs(p-t))),"plcc":corr(p,t),
            "srcc":corr(rankdata(p,method="average"),rankdata(t,method="average"))}


def keys(values): return list(zip(values["asset_ids"].astype(str),values["attacks"].astype(str),values["levels"].astype(str)))


def load_split(feature_dir,target_dir,split):
    with np.load(feature_dir/f"local_patch_features_{split}.npz") as f: features={k:f[k].copy() for k in f.files}
    with np.load(target_dir/f"local_patch_targets_{split}.npz") as t: targets={k:t[k].copy() for k in t.files}
    lookup={key:index for index,key in enumerate(keys(targets))}; missing=[key for key in keys(features) if key not in lookup]
    if missing: raise ValueError(f"Missing {len(missing)} targets in {split}")
    order=np.asarray([lookup[key] for key in keys(features)])
    for name in ("patch_geometry_quality","patch_texture_quality","patch_overall_quality","texture_supervision_mask"):
        features[name]=targets[name][order]
    return features


def statistics(train):
    patch=train["patch_mask"]
    geometry=np.concatenate([train["patch_descriptors"],
        np.log1p(16*train["patch_area"]/np.maximum(train["patch_area"].sum(1,keepdims=True),1e-12))[...,None]],2)
    view_mask=train["patch_view_mask"] & patch[:,:,None]; atlas_mask=train["patch_atlas_mask"] & patch
    def stats(values,mask):
        selected=values[mask]; return selected.mean(0).astype(np.float32),np.maximum(selected.std(0),1e-5).astype(np.float32)
    return {"geometry":stats(geometry,patch),"view_stats":stats(train["patch_view_stats"],view_mask),
            "atlas_stats":stats(train["patch_atlas_stats"],atlas_mask)}


def tensorize_features(raw,normalization,device):
    area=np.log1p(16*raw["patch_area"]/np.maximum(raw["patch_area"].sum(1,keepdims=True),1e-12))[...,None]
    geometry=np.concatenate([raw["patch_descriptors"],area],2)
    normalize=lambda x,name:(x-normalization[name][0])/normalization[name][1]
    return {"geometry":torch.from_numpy(normalize(geometry,"geometry")).float().to(device),
        "view_tokens":torch.from_numpy(raw["patch_view_tokens"].astype(np.float32)).to(device),
        "view_stats":torch.from_numpy(normalize(raw["patch_view_stats"],"view_stats")).float().to(device),
        "view_mask":torch.from_numpy(raw["patch_view_mask"]).bool().to(device),
        "atlas_tokens":torch.from_numpy(raw["patch_atlas_tokens"].astype(np.float32)).to(device),
        "atlas_stats":torch.from_numpy(normalize(raw["patch_atlas_stats"],"atlas_stats")).float().to(device),
        "atlas_mask":torch.from_numpy(raw["patch_atlas_mask"]).bool().to(device),
        "patch_mask":torch.from_numpy(raw["patch_mask"]).bool().to(device)}


def tensorize(raw,normalization,device):
    output=tensorize_features(raw,normalization,device)
    output.update({
        "geometry_target":torch.from_numpy(raw["patch_geometry_quality"]).float().to(device),
        "texture_target":torch.from_numpy(raw["patch_texture_quality"]).float().to(device),
        "overall_target":torch.from_numpy(raw["patch_overall_quality"]).float().to(device),
        "texture_supervision":torch.from_numpy(raw["texture_supervision_mask"]).bool().to(device),
        "attacks":raw["attacks"].astype(str)})
    return output


class LocalPatchHead(nn.Module):
    def __init__(self,use_atlas=True,geometry_context=False):
        super().__init__(); self.use_atlas=use_atlas; self.geometry_context=geometry_context
        self.geometry=nn.Sequential(nn.LayerNorm(59),nn.Linear(59,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        if geometry_context:
            self.geometry_attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
            self.geometry_norm=nn.LayerNorm(128)
        self.view=nn.Sequential(nn.LayerNorm(588),nn.Linear(588,128),nn.GELU())
        if use_atlas: self.atlas=nn.Sequential(nn.LayerNorm(588),nn.Linear(588,128),nn.GELU())
        self.texture=nn.Sequential(nn.Linear(128*(1+int(use_atlas)),128),nn.LayerNorm(128),nn.GELU())
        self.geometry_quality=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
        self.texture_quality=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())

    def forward(self,data,index=None):
        get=lambda name:data[name] if index is None else data[name][index]
        geometry=self.geometry(get("geometry"))
        if self.geometry_context:
            attended,_=self.geometry_attention(geometry,geometry,geometry,
                key_padding_mask=~get("patch_mask"),need_weights=False)
            geometry=self.geometry_norm(geometry+attended)
        view=self.view(torch.cat([get("view_tokens"),get("view_stats")],-1))
        mask=get("view_mask") & get("patch_mask")[:,:,None]; view=(view*mask[:,:,:,None]).sum(2)/mask.sum(2,keepdim=True).clamp_min(1)
        textures=[view]
        if self.use_atlas:
            atlas=self.atlas(torch.cat([get("atlas_tokens"),get("atlas_stats")],-1))
            textures.append(atlas*get("atlas_mask")[:,:,None])
        texture=self.texture(torch.cat(textures,-1))
        g=self.geometry_quality(geometry).squeeze(-1); t=self.texture_quality(texture).squeeze(-1)
        return {"geometry":g,"texture":t,"overall":torch.minimum(g,t)}


def ranking(pred,target,mask):
    delta=target[:,:,None]-target[:,None,:]; valid=mask[:,:,None]&mask[:,None,:]&(torch.abs(delta)>.05)
    if not valid.any(): return pred.new_zeros(())
    pd=pred[:,:,None]-pred[:,None,:]
    return F.softplus(-5*torch.sign(delta[valid])*pd[valid]).mean()


def loss_value(out,data,index):
    patch=data["patch_mask"][index]; texture=patch&data["texture_supervision"][index]
    losses=F.smooth_l1_loss(out["geometry"][patch],data["geometry_target"][index][patch])
    losses+=F.smooth_l1_loss(out["texture"][texture],data["texture_target"][index][texture])
    losses+=1.5*F.smooth_l1_loss(out["overall"][patch],data["overall_target"][index][patch])
    losses+=.1*(ranking(out["geometry"],data["geometry_target"][index],patch)+
                ranking(out["texture"],data["texture_target"][index],texture)+
                ranking(out["overall"],data["overall_target"][index],patch))
    return losses


@torch.no_grad()
def evaluate(model,data):
    model.eval(); out=model(data); patch=data["patch_mask"].cpu().numpy(); attacks=data["attacks"]
    prediction={k:v.cpu().numpy() for k,v in out.items()}
    truth={k:data[f"{k}_target"].cpu().numpy() for k in out}
    masks={"geometry":patch&np.isin(attacks,list(GEOMETRY))[:,None],
           "texture":patch&data["texture_supervision"].cpu().numpy()&np.isin(attacks,list(TEXTURE))[:,None],
           "overall":patch}
    return {name:metrics(prediction[name],truth[name],masks[name]) for name in prediction}


def train_variant(name,use_atlas,geometry_context,train,val,args,device):
    seed_all(args.seed); model=LocalPatchHead(use_atlas,geometry_context).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=None; best_state=None; best_epoch=None; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); order=torch.randperm(len(train["geometry"]),device=device); losses=[]
        for start in range(0,len(order),args.batch_size):
            index=order[start:start+args.batch_size]; out=model(train,index); loss=loss_value(out,train,index)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        result=evaluate(model,val); mean_srcc=np.mean([result[k]["srcc"] for k in ("geometry","texture","overall")])
        mean_mae=np.mean([result[k]["mae"] for k in ("geometry","texture","overall")]); key=(mean_srcc,-mean_mae,-epoch)
        history.append({"epoch":epoch,"loss":float(np.mean(losses)),"val":result,"selection_key":[float(mean_srcc),-float(mean_mae),-epoch]})
        if best is None or key>best: best=key; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
        if epoch==1 or epoch%10==0: print(name,epoch,float(np.mean(losses)),mean_srcc,flush=True)
    model.load_state_dict(best_state); return model,{"variant":name,"best_epoch":best_epoch,"val":evaluate(model,val),"history":history}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--feature-dir",type=Path,required=True)
    parser.add_argument("--target-dir",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=60); parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--lr",type=float,default=3e-4); parser.add_argument("--seed",type=int,default=2026); parser.add_argument("--device",default="cuda")
    parser.add_argument("--variants",nargs="+",choices=("view_only","view_atlas","view_atlas_geometry_context"),
                        default=("view_only","view_atlas","view_atlas_geometry_context"))
    args=parser.parse_args(); device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    raw={split:load_split(args.feature_dir,args.target_dir,split) for split in ("train","val")}; norm=statistics(raw["train"])
    data={split:tensorize(raw[split],norm,device) for split in raw}; args.output_dir.mkdir(parents=True,exist_ok=True)
    results={}
    candidates=(("view_only",False,False),("view_atlas",True,False),("view_atlas_geometry_context",True,True))
    for name,use_atlas,geometry_context in (item for item in candidates if item[0] in args.variants):
        model,result=train_variant(name,use_atlas,geometry_context,data["train"],data["val"],args,device); results[name]=result
        torch.save({"model":model.state_dict(),"use_atlas":use_atlas,"geometry_context":geometry_context,
                    "normalization":{k:[x.tolist() for x in v] for k,v in norm.items()},
                    "seed":args.seed,"feature_metadata":json.loads((args.feature_dir/"metadata.json").read_text()),
                    "target_metadata":json.loads((args.target_dir/"metadata.json").read_text()),"val":result["val"]},args.output_dir/f"{name}.pt")
    selected=max(results,key=lambda name:(np.mean([results[name]["val"][k]["srcc"] for k in ("geometry","texture","overall")]),
                                          -np.mean([results[name]["val"][k]["mae"] for k in ("geometry","texture","overall")])))
    report={"schema_version":1,"seed":args.seed,"selection":"Val mean Patch SRCC, then mean MAE",
            "selected":selected,"results":results,"test_blind_loaded":False}
    (args.output_dir/"results.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({"selected":selected,"val":results[selected]["val"]}))


if __name__=="__main__": main()
