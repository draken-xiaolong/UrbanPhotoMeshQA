"""Small pilot audit of achieved strengths, locality and repeated outputs."""
import argparse,json
from pathlib import Path
import numpy as np
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.process_degradations import SurfaceRegion,deform
from build_iteration3_pilot import stable_seed

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
records=json.loads((a.root/'manifest.json').read_text())['records']
clean=GltfReader(next(r['gltf_path'] for r in records if r['variant']=='clean')).load_mesh(include_texture=True)
labels=np.load(a.root/'source_patch_layout.npz')['face_patch']
result={'note':'Physical generation diagnostics, not MOS; target scale is not an assigned label.', 'types':{}}
for r in records:
    if '_level' not in r['variant']:continue
    op=r['operations'][0];kind=op['category'];row={'level':op['level'],'parameter':op['value'],
        'actual_faces':r['face_count'],'actual_face_ratio':r['face_count']/len(clean.faces),
        'content_digest':r['content_digest'],'zero_area_faces':r['zero_area_faces']}
    anchor=op.get('anchor_face')
    row['anchor_patch']=int(labels[anchor])+1 if anchor is not None else None
    with np.load(a.root/'assets'/r['variant']/'face_support.npz') as s:
        selected=s['geometry'] | s['texture']
        row['affected_patches']=sorted((np.unique(labels[selected])+1).tolist())
    if kind=='geometry_smoothing':
        mask=SurfaceRegion(clean,0,anchor).mask(.35)
        m=deform(clean,mask,.08,stable_seed(r['asset_id'],'surface-process'),smooth_steps=12,max_displacement=op['value'])
        row['actual_max_vertex_displacement']=float(np.linalg.norm(m.vertices-clean.vertices,axis=1).max())
        assert row['actual_max_vertex_displacement']<=op['value']+1e-7
    result['types'].setdefault(kind,[]).append(row)
for kind,rows in result['types'].items():
    assert len(rows)==4
    assert len({r['content_digest'] for r in rows})==4, f'Repeated levels: {kind}'
result['distinct_local_anchor_patches']=sorted({r['anchor_patch'] for rows in result['types'].values() for r in rows if r['anchor_patch'] is not None})
result['all_four_levels_distinct']=True
(a.root/'strength_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
