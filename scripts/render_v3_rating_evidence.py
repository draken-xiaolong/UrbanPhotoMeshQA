"""Seven-view evidence per variant, grouped contact sheets; no score inference."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw
from review_v3_visual_candidates import render

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--size',type=int,default=512)
p.add_argument('--sheets-only',action='store_true')
a=p.parse_args();out=a.root/'visual_review_v2';out.mkdir(exist_ok=True)
records=json.loads((a.root/'manifest.json').read_text())['records']
groups={}
for r in records:
    variant=r['variant'];target=out/variant
    if a.sheets_only and not (target/'views.jpg').exists():continue
    if not (target/'views.jpg').exists():render(Path(r['gltf_path']),target,a.size)
    key=variant.rsplit('_level',1)[0] if '_level' in variant else ('combined' if variant.startswith('combined') else 'clean')
    groups.setdefault(key,[]).append(variant)
    print(variant,flush=True)
for key,variants in groups.items():
    for start in range(0,len(variants),4):
        selected=variants[start:start+4];sheet=Image.new('RGB',(7*224,len(selected)*248),'white')
        for row,variant in enumerate(selected):
            ImageDraw.Draw(sheet).text((4,row*248+4),variant,fill='black')
            for v in range(7):
                im=Image.open(out/variant/f'view{v}.png');im.thumbnail((224,224))
                sheet.paste(im,(v*224,row*248+24))
        sheet.save(out/f'{key}_{start//4}.jpg',quality=95)
(out/'protocol.json').write_text(json.dumps({'size':a.size,'views':7,'render':'Y-up corrected, native glTF UV corrected; unlit deterministic rasterizer',
    'limitation':'Static views plus detail inspection; does not certify all occluded surfaces or replace interactive human confirmation.'},indent=2))
