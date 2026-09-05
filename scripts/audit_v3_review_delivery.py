"""Small delivery audit: finite attributes, dependencies, explicit AI opinions, duplicates."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.integrity import asset_digest

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
rows=json.loads((a.root/'manifest.json').read_text())['records']
scores=json.loads((a.root/'machine_scores.json').read_text())['scores']
groups=defaultdict(list);dependencies=0
for r in rows:
    path=Path(r['gltf_path']);digest,n=asset_digest(path)
    assert digest==r['asset_digest'],r['variant'];dependencies+=len(n)
    m=GltfReader(path).load_mesh(include_texture=True)
    assert np.isfinite(m.vertices).all() and np.isfinite(m.normals).all()
    for material in np.unique(m.face_materials):
        if m.metadata['material_texture_paths'][material]:
            assert np.isfinite(m.texcoords[m.faces[m.face_materials==material]]).all(),r['variant']
    assert scores[r['variant']]['digest']==r['content_digest']
    assert scores[r['variant']]['method']=='ai_visual_review_v2'
    groups[r['content_digest']].append(r['variant'])
result={'versions':len(rows),'unique_decoded_versions':len(groups),
        'duplicate_groups':[g for g in groups.values() if len(g)>1],
        'dependency_references_checked':dependencies,'missing_dependencies':0,
        'finite_geometry_and_textured_uv':True,'ai_visual_scores':len(scores),
        'scale_counts':dict(sorted(Counter(s['scale'] for s in scores.values()).items())),
        'formal_training_started':False,'human_confirmation':'pending',
        'note':'Duplicates and generation-realism flags require review before any formal dataset freeze.'}
(a.root/'review_delivery_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(a.root.parent.name,result)
