"""Stage only the two visually inspected 2026-09-06 pilot candidates."""
import json
import argparse
import subprocess
import sys
from pathlib import Path
from build_iteration3_pilot import stage_clean, validate

root=Path('/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/HK3D-Individualised-V3')
choices={
    'B415722108801063A0':('新候选A · 浅色住宅塔楼','各立面窗格可辨、轮廓稳定、屋顶清楚；底座局部灰面和立面色彩分区需用户复核。'),
    'B421142104901063A0':('新候选B · 深色阳台住宅','多面阳台和窗格连续、屋顶完整，未见大片涂抹；局部细纹理偏软、细小栏杆仅贴图表达。'),
}
rows=json.loads(Path('artifacts/manifests/iteration2_source_audit_seed2026.json').read_text())['records']
p=argparse.ArgumentParser();p.add_argument('--only',nargs='*',choices=list(choices));args=p.parse_args()
for r in rows:
    if r['asset_id'] not in choices:continue
    if args.only and r['asset_id'] not in args.only:continue
    assert r['split']=='train'
    folder=root/'assets'/r['asset_id'];destination=folder/'clean'
    source=Path('/Volumes/SANDISK-ELE/HK3D-Individualised')/r['source_gltf']
    if destination.exists():
        assert json.loads((destination/'metadata.json').read_text())['source']==str(source)
        path=destination/'clean_scale5.gltf'
    else:path=stage_clean(source,destination)
    name,reason=choices[r['asset_id']]
    metadata={'asset_id':r['asset_id'],'display_name':name,'source':str(source),
              'human_confirmed':False,'clean_admission':'ai_visual_pass_pending_user',
              'suggested_scale':5,'review_reason':reason,
              'evidence':str(root/'_review/clean_reselection_20260906'/r['asset_id']),
              'review_method':'seven Y-up-corrected 896px views plus full-resolution facade inspection',
              **validate(path)}
    (destination/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(metadata,flush=True)
    subprocess.run([sys.executable,'scripts/build_v3_process_pilot.py','--asset-root',str(folder),'--resume'],check=True)
    subprocess.run([sys.executable,'scripts/audit_v3_process_pilot.py','--root',str(folder/'process_v2')],check=True)
