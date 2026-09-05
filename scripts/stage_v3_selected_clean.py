"""Stage the two visually reviewed additional calibration buildings."""
import json
from pathlib import Path
from build_iteration3_pilot import stage_clean,validate
root=Path('/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/HK3D-Individualised-V3')
choices={'B360381775901063A0':'规整办公楼','B354401777301063A0':'狭长转角建筑'}
rows=json.loads((root/'previews/clean_shortlist/candidates.json').read_text())
for r in rows:
    if r['asset_id'] not in choices:continue
    destination=root/'assets'/r['asset_id']/'clean'
    if destination.exists():raise FileExistsError(destination)
    source=Path('/Volumes/SANDISK-ELE/HK3D-Individualised')/r['source_gltf']
    path=stage_clean(source,destination)
    metadata={'asset_id':r['asset_id'],'display_name':choices[r['asset_id']],
              'source':str(source),'selection':'12 training-pool candidates inspected in three views; provisional clean anchor',
              'human_confirmed':False,**validate(path)}
    (destination/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(metadata)
