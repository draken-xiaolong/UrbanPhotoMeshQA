"""Current-face/Patch intervention evidence from saved pixel masks; assets stay unchanged."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.integrity import sha256_file
from urbanphotomeshqa.uv_support import pixel_support_on_faces, patch_support


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if str(args.output).startswith('/Volumes/') and not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('External disk not mounted')
    manifest=args.root/'candidate_manifest.json'
    rows=json.loads(manifest.read_text())['records']
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for row in rows:
        folder=(args.root/row['gltf']).parent
        output=args.output/(row['variant_id']+'.json')
        bindings={n:sha256_file(folder/n) for n in ('intervention_support.npz','current_patch_layout.npz','metadata.json')}
        if output.exists():
            previous=json.loads(output.read_text())
            if previous['bindings']!=bindings:raise ValueError('Changed inputs: use a new audit revision')
            results.append(previous);continue
        mesh=GltfReader(args.root/row['gltf']).load_mesh(include_texture=True)
        tri=mesh.vertices[mesh.faces]
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        with np.load(folder/'current_patch_layout.npz') as layout:
            face_patch=layout['face_patch']
        report={'variant_id':row['variant_id'],'content_digest':row['content_digest'],'bindings':bindings,
                'formal_admitted':False,'label_source':'intervention_support_only',
                'visible_labels_created':0,'local_quality_labels_created':0,'texture_operations':[],
                'method':'native_gltf_uv_area_uniform_quadrature_256_and_1024_v1',
                'limitations':['Numerical surface-area approximation, not exact pixel/triangle intersection',
                    'No visibility/occlusion or perceptual abnormality claim; unknown materials remain unknown',
                    'No source/current geometric correspondence is inferred for QEM']}
        textures=mesh.metadata.get('material_texture_paths',[])
        shared={x for x in textures if x and textures.count(x)>1}
        report['shared_texture_materials_require_extra_audit']=bool(shared)
        with np.load(folder/'intervention_support.npz') as evidence:
            numbers=sorted({int(m.group(1)) for key in evidence.files if (m:=re.fullmatch(r'pixel_support_op(\d+)_material\d+',key))})
            for number in numbers:
                masks={}
                for key in evidence.files:
                    m=re.fullmatch(r'pixel_support_op'+str(number)+r'_material(\d+)',key)
                    if m:
                        material=int(m.group(1));changed=f'pixel_changed_op{number}_material{material}'
                        masks[material]=evidence[changed] if changed in evidence else evidence[key]
                # Propagate through all materials referring to the same final texture image.
                for material in range(len(textures)):
                    same=[v for k,v in masks.items() if textures[material] and k<len(textures) and textures[k]==textures[material]]
                    if same and len({x.shape for x in same})==1:masks[material]=np.logical_or.reduce(same)
                coarse=pixel_support_on_faces(mesh,masks,256)
                fine=pixel_support_on_faces(mesh,masks,1024)
                delta=np.abs(fine-coarse)
                aggregate=patch_support(mesh,face_patch,fine)
                known=np.isfinite(fine)
                item={'operation_index':number,'known_surface_fraction':float(area[known].sum()/area.sum()),
                    'supported_surface_fraction':float(np.sum(area[known]*fine[known])/area.sum()),
                    'quadrature_disagreement_area_mean':float(np.sum(area[known]*delta[known])/area[known].sum()) if known.any() else None,
                    'face_fraction_file':f"{row['variant_id']}_op{number}.npz",
                    'patches':[{k:float(v[i]) if np.isfinite(v[i]) else None for k,v in aggregate.items()} for i in range(16)]}
                np.savez_compressed(args.output/item['face_fraction_file'],face_fraction=fine,
                    quadrature_delta=delta,face_patch=face_patch,known=known)
                item['sha256']=sha256_file(args.output/item['face_fraction_file'])
                report['texture_operations'].append(item)
        with output.open('x') as f:json.dump(report,f,indent=2)
        results.append(report)
        print(row['variant_id'],'current UV support audited; visible labels remain unknown',flush=True)
    (args.output/'summary.json').write_text(json.dumps({'manifest_sha256':sha256_file(manifest),
        'count':len(results),'formal_admitted':0,'reports':[r['variant_id']+'.json' for r in results]},indent=2))


if __name__=='__main__':main()
