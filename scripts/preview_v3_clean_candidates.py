"""Small visual shortlist from previously audited TRAIN assets only."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.texture import render_textured_view

root=Path('/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/HK3D-Individualised-V3')
out=root/'previews'/'clean_shortlist';out.mkdir(parents=True,exist_ok=True)
rows=json.loads(Path('artifacts/manifests/iteration2_source_audit_seed2026.json').read_text())['records']
rows=[r for r in rows if r['split']=='train' and 800<r['face_count']<8000 and r['texture_pixels']<14000000]
rows=sorted(rows,key=lambda r:-r['face_count'])[:12]
sheet=Image.new('RGB',(720,len(rows)*260),'white')
for i,r in enumerate(rows):
    path=Path('/Volumes/SANDISK-ELE/HK3D-Individualised')/r['source_gltf']
    m=GltfReader(path).load_mesh(include_texture=True)
    uv=m.texcoords.copy();uv[:,1]=1-uv[:,1];m=replace(m,texcoords=uv)
    ImageDraw.Draw(sheet).text((6,i*260+4),r['asset_id'],fill='black')
    for j,d in enumerate([(1,.6,1),(-1,.6,-1),(0,1,.01)]):
        image=Image.fromarray(render_textured_view(m,direction=d,size=240));sheet.paste(image,(j*240,i*260+20))
    print(r['asset_id'],np.ptp(m.vertices,axis=0).tolist(),flush=True)
sheet.save(out/'shortlist.jpg')
(out/'candidates.json').write_text(json.dumps(rows,indent=2))
