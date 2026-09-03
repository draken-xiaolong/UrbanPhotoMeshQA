#!/usr/bin/env python3
"""Train a quality-specific face GATv2 with strong texture features."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from urbanphotomeshqa.data import pad_mesh_graphs


ATTACKS = ("clean", "geometry_hole", "mesh_simplification_qem", "geometry_noise_spike",
           "texture_detail_loss", "texture_region_missing", "texture_misalignment")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def rankdata(values):
    values=np.asarray(values); order=np.argsort(values,kind="mergesort"); sorted_values=values[order]
    ranks=np.empty(len(values),np.float64); start=0
    while start<len(values):
        stop=start+1
        while stop<len(values) and sorted_values[stop]==sorted_values[start]: stop+=1
        ranks[order[start:stop]]=0.5*(start+stop-1); start=stop
    return ranks


def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else 0.0


class GATv2Layer(nn.Module):
    def __init__(self, input_dim, output_dim=128, heads=4):
        super().__init__(); self.heads=heads; self.width=output_dim//heads
        self.query=nn.Linear(input_dim,output_dim,bias=False)
        self.key=nn.Linear(input_dim,output_dim,bias=False)
        self.value=nn.Linear(input_dim,output_dim,bias=False)
        self.attention=nn.Parameter(torch.randn(heads,self.width)*0.02)
        self.residual=nn.Linear(input_dim,output_dim,bias=False) if input_dim!=output_dim else nn.Identity()
        self.norm=nn.LayerNorm(output_dim)

    def forward(self,x,neighbors,mask):
        b,n,_=x.shape; batch=torch.arange(b,device=x.device)[:,None,None]
        q=self.query(x).view(b,n,self.heads,self.width)
        k=self.key(x).view(b,n,self.heads,self.width)[batch,neighbors]
        v=self.value(x).view(b,n,self.heads,self.width)[batch,neighbors]
        logits=(F.leaky_relu(q[:,:,None]+k,0.2)*self.attention[None,None,None]).sum(-1)
        weights=torch.softmax(logits,dim=2)
        message=(weights[...,None]*v).sum(2).reshape(b,n,-1)
        return self.norm(self.residual(x)+message)*mask[:,:,None]


class QualityGAT(nn.Module):
    def __init__(self,point_dim,morph_dim,texture_dim):
        super().__init__()
        self.face_in=nn.Sequential(nn.LayerNorm(13),nn.Linear(13,96),nn.GELU())
        self.gat1=GATv2Layer(96,128,4); self.gat2=GATv2Layer(128,128,4)
        self.mesh=nn.Sequential(nn.Linear(128*2+8,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128))
        self.projections=nn.ModuleList([
            nn.Sequential(nn.LayerNorm(point_dim),nn.Linear(point_dim,128),nn.GELU()),
            nn.Sequential(nn.LayerNorm(morph_dim),nn.Linear(morph_dim,128),nn.GELU()),
            nn.Sequential(nn.LayerNorm(texture_dim),nn.Linear(texture_dim,128),nn.GELU()),
        ])
        self.embedding=nn.Parameter(torch.randn(1,4,128)*.02)
        self.cross=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
        self.shared=nn.Sequential(nn.Linear(512,384),nn.LayerNorm(384),nn.GELU(),nn.Dropout(.15),
                                  nn.Linear(384,256),nn.GELU())
        self.regression=nn.Sequential(nn.Linear(256,4),nn.Sigmoid())
        self.attack=nn.Linear(256,len(ATTACKS)); self.ordinal=nn.Linear(256,4)

    def forward(self,graph,point,morph,texture,texture_views=None):
        x=self.face_in(graph["face_features"]); x=self.gat1(x,graph["neighbors"],graph["mask"])
        x=self.gat2(x,graph["neighbors"],graph["mask"]); valid=graph["mask"][:,:,None]
        maximum=x.masked_fill(~valid,torch.finfo(x.dtype).min).max(1).values
        average=(x*valid).sum(1)/valid.sum(1).clamp_min(1)
        mesh=self.mesh(torch.cat([maximum,average,graph["topology"]],1))
        tokens=torch.stack([self.projections[0](point),mesh,self.projections[1](morph),
                            self.projections[2](texture)],1)+self.embedding
        attended,_=self.cross(tokens,tokens,tokens,need_weights=False)
        shared=self.shared((tokens+attended).flatten(1)); reg=self.regression(shared)
        return {"severity":reg[:,0],"overall":reg[:,1],"geometry":reg[:,2],"texture":reg[:,3],
                "attack":self.attack(shared),"ordinal":self.ordinal(shared)}


class HierarchicalQualityGAT(nn.Module):
    """Face GAT -> learned Patch tokens -> Building with bidirectional texture attention."""
    def __init__(self, point_dim, morph_dim, texture_dim):
        super().__init__()
        self.face_in=nn.Sequential(nn.LayerNorm(13),nn.Linear(13,96),nn.GELU())
        self.gat1=GATv2Layer(96,128,4); self.gat2=GATv2Layer(128,128,4)
        self.patch_queries=nn.Parameter(torch.randn(1,16,128)*.02)
        self.face_to_patch=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
        layer=nn.TransformerEncoderLayer(128,4,384,dropout=.15,activation="gelu",batch_first=True,norm_first=True)
        self.patch_transformer=nn.TransformerEncoder(layer,2)
        self.texture_projection=nn.Sequential(nn.LayerNorm(texture_dim),nn.Linear(texture_dim,128),nn.GELU())
        self.texture_transformer=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128,4,384,dropout=.1,activation="gelu",batch_first=True,norm_first=True),1)
        self.patch_to_texture=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
        self.texture_to_patch=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
        self.point=nn.Sequential(nn.LayerNorm(point_dim),nn.Linear(point_dim,128),nn.GELU())
        self.morph=nn.Sequential(nn.LayerNorm(morph_dim),nn.Linear(morph_dim,128),nn.GELU())
        self.geometry_pool=nn.Sequential(nn.Linear(256+8,256),nn.GELU(),nn.Linear(256,128))
        self.embedding=nn.Parameter(torch.randn(1,4,128)*.02)
        self.building_attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
        self.shared=nn.Sequential(nn.Linear(512,384),nn.LayerNorm(384),nn.GELU(),nn.Dropout(.2),nn.Linear(384,256),nn.GELU())
        self.regression=nn.Sequential(nn.Linear(256,4),nn.Sigmoid()); self.attack=nn.Linear(256,7); self.ordinal=nn.Linear(256,4)

    def forward(self,graph,point,morph,texture,texture_views=None):
        if texture_views is None: raise ValueError("Hierarchical model requires texture view tokens")
        face=self.gat2(self.gat1(self.face_in(graph["face_features"]),graph["neighbors"],graph["mask"]),graph["neighbors"],graph["mask"])
        query=self.patch_queries.expand(face.shape[0],-1,-1)
        patches,_=self.face_to_patch(query,face,face,key_padding_mask=~graph["mask"],need_weights=False)
        patches=self.patch_transformer(query+patches)
        texture_tokens=self.texture_transformer(self.texture_projection(texture_views))
        patch_cross,_=self.patch_to_texture(patches,texture_tokens,texture_tokens,need_weights=False)
        texture_cross,_=self.texture_to_patch(texture_tokens,patches,patches,need_weights=False)
        patches=patches+patch_cross; texture_tokens=texture_tokens+texture_cross
        geometry=self.geometry_pool(torch.cat([patches.mean(1),patches.max(1).values,graph["topology"]],1))
        tokens=torch.stack([self.point(point),geometry,self.morph(morph),texture_tokens.mean(1)],1)+self.embedding
        attended,_=self.building_attention(tokens,tokens,tokens,need_weights=False)
        shared=self.shared((tokens+attended).flatten(1)); reg=self.regression(shared)
        return {"severity":reg[:,0],"overall":reg[:,1],"geometry":reg[:,2],"texture":reg[:,3],
                "attack":self.attack(shared),"ordinal":self.ordinal(shared)}


def load_split(split,args):
    with np.load(args.feature_dir/f"features_{split}.npz") as z:
        base={name:z[name].copy() for name in z.files}
    with np.load(args.texture_dir/f"strong_texture_{split}.npz") as z:
        strong={name:z[name].copy() for name in z.files}
    with np.load(args.target_dir/f"objective_targets_{split}.npz") as z:
        target={name:z[name].copy() for name in z.files}
    keys=list(zip(base["asset_ids"].astype(str),base["attacks"].astype(str),base["levels"].astype(str)))
    for values in (strong,target):
        other=list(zip(values["asset_ids"].astype(str),values["attacks"].astype(str),values["levels"].astype(str)))
        if keys!=other: raise ValueError(f"Order mismatch: {split}")
    audit=json.loads(args.cache_audit.read_text(encoding="utf-8"))["records"]
    paths={(r["asset_id"],r["attack"],r["level"]):r["cache_path"] for r in audit}
    tiles={(r["asset_id"],r["attack"],r["level"]):r["sheet"] for r in audit}
    graphs=[]
    for index,item in enumerate(keys):
        with np.load(paths[item]) as z:
            graphs.append({name:z[name].copy() for name in ("face_features","neighbors","topology")})
        if (index+1)%500==0: print(f"load {split} {index+1}/{len(keys)}",flush=True)
    group_names=[f"{asset_id}::{attack}" if attack != "clean" else f"{asset_id}::clean::{i}"
                 for i,(asset_id,attack,_) in enumerate(keys)]
    group_lookup={name:i for i,name in enumerate(dict.fromkeys(group_names))}
    return {"keys":keys,"graphs":graphs,"faces":np.asarray([len(g["face_features"]) for g in graphs]),
            "tiles":np.asarray([tiles[item] for item in keys]),
            "levels":base["levels"].astype(str),
            "groups":np.asarray([group_lookup[name] for name in group_names],dtype=np.int64),
            "point":base["point_identity"].astype(np.float32),"morph":base["morphology"].astype(np.float32),
            "texture":strong["texture"].astype(np.float32),
            "texture_views":strong.get("texture_views", np.repeat(strong["texture"][:,None], 6, axis=1)).astype(np.float32),
            "attack":base["attack_index"].astype(np.int64),
            "severity":base["severity"].astype(np.float32),
            "overall":target["overall_quality"].astype(np.float32),
            "geometry":target["geometry_quality"].astype(np.float32),
            "texture_target":target["texture_quality"].astype(np.float32),
            "ordinal":target["ordinal_grade"].astype(np.int64)}


def batches(data,batch_size,shuffle,seed):
    order=np.argsort(data["faces"],kind="stable")
    chunks=[order[i:i+batch_size] for i in range(0,len(order),batch_size)]
    if shuffle: random.Random(seed).shuffle(chunks)
    return chunks


def grouped_batches(data,batch_size,shuffle,seed):
    """Keep severity variants together while roughly bucketing groups by mesh size."""
    members={}
    for index,group in enumerate(data["groups"]): members.setdefault(int(group),[]).append(index)
    groups=list(members.values())
    groups.sort(key=lambda indices: float(np.median(data["faces"][indices])))
    chunks=[]; current=[]
    for indices in groups:
        if current and len(current)+len(indices)>batch_size:
            chunks.append(np.asarray(current,dtype=np.int64)); current=[]
        current.extend(indices)
    if current: chunks.append(np.asarray(current,dtype=np.int64))
    if shuffle: random.Random(seed).shuffle(chunks)
    return chunks


def forward_batch(model,data,index,device):
    graph={k:v.to(device) for k,v in pad_mesh_graphs([data["graphs"][i] for i in index]).items()}
    tensors={k:torch.from_numpy(data[k][index]).to(device) for k in ("point","morph","texture","texture_views")}
    return model(graph,tensors["point"],tensors["morph"],tensors["texture"],tensors["texture_views"])


@torch.no_grad()
def predict(model,data,batch_size,device):
    model.eval(); outputs={
        "overall": np.empty(len(data["keys"]), np.float32),
        "geometry": np.empty(len(data["keys"]), np.float32),
        "texture": np.empty(len(data["keys"]), np.float32),
        "severity": np.empty(len(data["keys"]), np.float32),
        "attack": np.empty((len(data["keys"]), len(ATTACKS)), np.float32),
        "ordinal": np.empty((len(data["keys"]), 4), np.float32),
    }
    for index in batches(data,batch_size,False,0):
        out=forward_batch(model,data,index,device)
        for name in outputs: outputs[name][index] = out[name].cpu().numpy()
    output=outputs
    return output


def metrics_from_output(output, data):
    result={}
    for name,target in (("overall","overall"),("geometry","geometry"),("texture","texture_target"),("severity","severity")):
        result[name]={"mae":float(np.mean(np.abs(output[name]-data[target]))),
                      "plcc":corr(output[name],data[target]),
                      "srcc":corr(rankdata(output[name]),rankdata(data[target]))}
    prediction=output["attack"].argmax(1); f1=[]
    for label in range(len(ATTACKS)):
        tp=np.sum((prediction==label)&(data["attack"]==label)); fp=np.sum((prediction==label)&(data["attack"]!=label)); fn=np.sum((prediction!=label)&(data["attack"]==label))
        precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); f1.append(2*precision*recall/max(precision+recall,1e-12))
    result["macro_f1"]=float(np.mean(f1)); grade=1+(1/(1+np.exp(-output["ordinal"]))>=.5).sum(1)
    result["ordinal_mae"]=float(np.mean(np.abs(grade-data["ordinal"])))
    return result


@torch.no_grad()
def evaluate(model,data,batch_size,device):
    return metrics_from_output(predict(model, data, batch_size, device), data)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--feature-dir",type=Path,required=True)
    parser.add_argument("--texture-dir",type=Path,required=True); parser.add_argument("--target-dir",type=Path,required=True)
    parser.add_argument("--cache-audit",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=40); parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--lr",type=float,default=3e-4); parser.add_argument("--seed",type=int,default=2026); parser.add_argument("--device",default="cuda")
    parser.add_argument("--architecture",choices=("gatv2","hierarchical"),default="gatv2")
    parser.add_argument("--ranking-scope",choices=("all","same_group"),default="all")
    parser.add_argument("--quality-balance",choices=("none","clean_light"),default="none")
    args=parser.parse_args(); seed_all(args.seed); device=torch.device(args.device)
    train=load_split("train",args); val=load_split("val",args)
    stats={}
    for name in ("point","morph","texture"):
        mean=train[name].mean(0); std=np.maximum(train[name].std(0),1e-5); stats[name]=(mean,std)
        train[name]=(train[name]-mean)/std; val[name]=(val[name]-mean)/std
    view_mean=train["texture_views"].reshape(-1,train["texture_views"].shape[-1]).mean(0)
    view_std=np.maximum(train["texture_views"].reshape(-1,train["texture_views"].shape[-1]).std(0),1e-5)
    train["texture_views"]=(train["texture_views"]-view_mean)/view_std; val["texture_views"]=(val["texture_views"]-view_mean)/view_std
    model_class=QualityGAT if args.architecture=="gatv2" else HierarchicalQualityGAT
    model=model_class(train["point"].shape[1],train["morph"].shape[1],train["texture"].shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    weights=torch.from_numpy(len(train["attack"])/np.maximum(np.bincount(train["attack"],minlength=7),1)).float().to(device); weights/=weights.mean()
    best=None; best_state=None; best_epoch=0; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        iterator=grouped_batches(train,args.batch_size,True,args.seed+epoch) if args.ranking_scope=="same_group" else batches(train,args.batch_size,True,args.seed+epoch)
        for index in iterator:
            out=forward_batch(model,train,index,device); idx=np.asarray(index)
            attack=torch.from_numpy(train["attack"][idx]).to(device); severity=torch.from_numpy(train["severity"][idx]).to(device)
            overall=torch.from_numpy(train["overall"][idx]).to(device); geometry=torch.from_numpy(train["geometry"][idx]).to(device); texture=torch.from_numpy(train["texture_target"][idx]).to(device)
            ordinal=torch.from_numpy(train["ordinal"][idx]).to(device); ordinal_truth=(ordinal[:,None]>torch.arange(1,5,device=device)[None]).float()
            sample_weight=torch.ones_like(overall)
            if args.quality_balance=="clean_light":
                level=train["levels"][idx]
                sample_weight=torch.from_numpy(np.where(level=="clean",4.0,np.where(level=="light",1.5,1.0)).astype(np.float32)).to(device)
            def weighted_smooth_l1(prediction,target):
                return (F.smooth_l1_loss(prediction,target,reduction="none")*sample_weight).sum()/sample_weight.sum()
            loss=.5*F.cross_entropy(out["attack"],attack,weight=weights)+.5*F.smooth_l1_loss(out["severity"],severity)
            loss+=3*weighted_smooth_l1(out["overall"],overall)+1.5*weighted_smooth_l1(out["geometry"],geometry)+1.5*weighted_smooth_l1(out["texture"],texture)
            ordinal_loss=F.binary_cross_entropy_with_logits(out["ordinal"],ordinal_truth,reduction="none").mean(1)
            loss+=.3*(ordinal_loss*sample_weight).sum()/sample_weight.sum()
            delta=overall[:,None]-overall[None]; valid=delta.abs()>.05
            if args.ranking_scope=="same_group":
                group=torch.from_numpy(train["groups"][idx]).to(device)
                valid &= group[:,None].eq(group[None])
            if valid.any(): loss+=.2*F.softplus(-5*torch.sign(delta[valid])*(out["overall"][:,None]-out["overall"][None])[valid]).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); losses.append(float(loss))
        metrics=evaluate(model,val,args.batch_size,device); history.append({"epoch":epoch,"loss":float(np.mean(losses)),"val":metrics})
        score=(metrics["overall"]["srcc"],-metrics["overall"]["mae"])
        if best is None or score>best: best=score; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"epoch={epoch} loss={np.mean(losses):.4f} val_srcc={metrics['overall']['srcc']:.4f} val_mae={metrics['overall']['mae']:.4f} f1={metrics['macro_f1']:.4f}",flush=True)
    model.load_state_dict(best_state); result=evaluate(model,val,args.batch_size,device); args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save({"schema_version":1,"seed":args.seed,"architecture":args.architecture,
                "model":model.state_dict(),"statistics":{k:{"mean":v[0].tolist(),"std":v[1].tolist()} for k,v in stats.items()},
                "texture_view_statistics":{"mean":view_mean.tolist(),"std":view_std.tolist()}},args.output_dir/"gatv2_quality.pt")
    payload={"schema_version":1,"status":"COMPLETE","seed":args.seed,"architecture":args.architecture,"ranking_scope":args.ranking_scope,"quality_balance":args.quality_balance,"best_epoch":best_epoch,"val":result,"history":history,"protocol":"Train/Val only; Test/Blind not loaded"}
    (args.output_dir/"results.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"best_epoch":best_epoch,"val":result},ensure_ascii=False))


if __name__=="__main__": main()
