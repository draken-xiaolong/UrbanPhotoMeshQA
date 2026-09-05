"""Bounded, reproducible visual evidence; no automatic quality decisions."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.texture import render_textured_view

ROOT = Path('/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/HK3D-Individualised-V3')
DIRECTIONS = [(1,0,.15),(-1,0,.15),(0,1,.15),(0,-1,.15),(0,0,1),(1,1,.7),(-1,-1,.7)]

def render(path, destination, size=640, directions=DIRECTIONS):
    destination.mkdir(parents=True, exist_ok=True)
    mesh = GltfReader(path).load_mesh(include_texture=True)
    uv = mesh.texcoords.copy(); uv[:,1] = 1-uv[:,1]
    # glTF is Y-up, while this legacy diagnostic renderer is Z-up.
    # Rotation is restricted to the disposable render copy, never source assets.
    vertices=mesh.vertices[:,[0,2,1]].copy();vertices[:,1]*=-1
    mesh = replace(mesh, texcoords=uv, vertices=vertices)
    sheet = Image.new('RGB', (size*2,size*4),'white')
    for i,d in enumerate(directions):
        im = Image.fromarray(render_textured_view(mesh,direction=d,size=size))
        # Renderer image coordinates increase upward; correct display orientation.
        im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        im.save(destination/f'view{i}.png')
        sheet.paste(im,((i%2)*size,(i//2)*size))
    sheet.save(destination/'views.jpg',quality=95)
    return {'vertices':len(mesh.vertices),'faces':len(mesh.faces),'extent':np.ptp(mesh.vertices,axis=0).tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');p.add_argument('--offset',type=int,default=0)
    p.add_argument('--count',type=int,default=12);p.add_argument('--size',type=int,default=400)
    a=p.parse_args()
    rows=json.loads(Path('artifacts/manifests/iteration2_source_audit_seed2026.json').read_text())['records']
    rows=[r for r in rows if r['split']=='train' and r['status']=='qualified']
    if a.ids: rows=[r for r in rows if r['asset_id'] in a.ids]
    else:
        old={d.name for d in (ROOT/'assets').iterdir()}
        rows=[r for r in rows if r['asset_id'] not in old and 1500<r['face_count']<16000]
        rows=sorted(rows,key=lambda r:-r['face_count'])[a.offset:a.offset+a.count]
    out=ROOT/'_review'/'clean_reselection_20260906';out.mkdir(parents=True,exist_ok=True)
    sheet=Image.new('RGB',(a.size*3,(a.size+24)*len(rows)),'white')
    for i,r in enumerate(rows):
        source=Path('/Volumes/SANDISK-ELE/HK3D-Individualised')/r['source_gltf']
        dst=out/r['asset_id']
        directions=DIRECTIONS if a.ids else [DIRECTIONS[5],DIRECTIONS[6],DIRECTIONS[4]]
        r['visual_stats']=render(source,dst,a.size,directions)
        ImageDraw.Draw(sheet).text((4,i*(a.size+24)),r['asset_id'],fill='black')
        for j in range(3):sheet.paste(Image.open(dst/f'view{j}.png'),(j*a.size,i*(a.size+24)+24))
        print(r['asset_id'],r['visual_stats'],flush=True)
    (out/f'candidates_{a.offset}.json').write_text(json.dumps(rows,indent=2))
    sheet.save(out/f'shortlist_{a.offset}.jpg',quality=93)

if __name__=='__main__':main()
