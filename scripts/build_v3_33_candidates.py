"""Six-class candidate generator. Output is ungraded and never a formal dataset.

Run on GPU host for production. A content-bound scale5 Clean admission is required.
Each invocation creates one new building revision; failed runs remain recoverable.
"""
import argparse
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from build_iteration3_pilot import export_geometry, stage_clean, validate
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.integrity import asset_digest
from urbanphotomeshqa.patches import topological_patch_layout
from urbanphotomeshqa.process_degradations import MultiSurfaceRegion, areas, deform, local_texture, surface_mask, textured_qem, island_projection_ghost
from urbanphotomeshqa.v3_protocol import PATCH_COUNT, VERSION, effective_rating, planned_slots, stable_seed


def recipe(slot, building_index, diagonal):
    """Development starting points, not calibrated guarantees or label rules."""
    level = 5-slot['target_scale']
    if level == 0:
        return []
    i = level-1
    result = []
    for category in slot['applied_classes']:
        entry = {'class':category, 'level':level}
        if category == 'G1':
            entry.update(kind='missing_faces', fraction=[.025,.10,.25,.45][i])
        elif category == 'G2':
            entry.update(kind='correlated_noise', fraction=[.20,.35,.55,.75][i],
                         amplitude=diagonal*[.002,.006,.015,.035][i])
        elif category == 'G3':
            if building_index % 2:
                entry.update(kind='bounded_smoothing', fraction=[.25,.45,.65,.85][i],
                             displacement_cap=diagonal*[.001,.004,.012,.03][i])
            else:
                entry.update(kind='textured_qem', retained=[.80,.60,.35,.12][i])
        elif category == 'T1':
            entry.update(kind='downsample_encoding', ratio=[.25,.08,.025,.008][i],
                         jpeg_quality=[55,35,20,10][i])
        elif category == 'T2':
            entry.update(kind='projection_ghost', fraction=[.25,.45,.65,.85][i],
                         shift_relative=[.08,.18,.32,.48][i],
                         sampling_version='raster_connected_uv_domain_v2',
                         boundary_policy='nearest_valid_same_component',
                         limitation='touching UV islands may share a raster component; not camera-grounded')
        elif category == 'T3':
            radiometric = slot['variant_id']=='C5' or (slot['variant_id'].startswith('T3') and building_index%2)
            entry.update(kind='radiometric' if radiometric else 'missing_texture',
                         fraction=([.25,.45,.65,.85] if radiometric else [.008,.05,.18,.50])[i],
                         strength=[.5,1.,1.8,3.5][i])
        result.append(entry)
    return result


def decoded_digest(mesh):
    digest = hashlib.sha256()
    for name in ('vertices','faces','texcoords','normals','face_materials'):
        array = np.ascontiguousarray(getattr(mesh,name))
        digest.update(str((name,array.dtype.str,array.shape)).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(mesh.metadata.get('material_profiles',[]),sort_keys=True,default=str).encode())
    for texture in mesh.metadata.get('material_texture_paths',[]):
        if texture:
            with Image.open(texture) as image:
                digest.update(str(image.size).encode())
                digest.update(image.convert('RGBA').tobytes())
    return digest.hexdigest()


def build(source, root, admission_path, building_index, selected=None):
    source_digest, _ = asset_digest(source)
    admission = json.loads(admission_path.read_text())
    rating = effective_rating(admission.get('ratings',{}),source_digest)
    if rating is None or rating['scale'] != 5 or admission.get('technical_valid') is not True:
        raise ValueError('Content-bound independent scale5 Clean admission required')
    if root.exists():
        raise FileExistsError('Use a new revision; existing trials are never overwritten')
    clean = GltfReader(source).load_mesh(include_texture=True)
    area = areas(clean)
    if (area>0).sum() < PATCH_COUNT or not np.isfinite(clean.vertices).all():
        raise ValueError('Clean geometry is not eligible')
    diagonal = float(np.linalg.norm(np.ptp(clean.vertices[clean.faces].reshape(-1,3),axis=0)))
    root.mkdir(parents=True)
    scratch = root / '_scratch'
    scratch.mkdir()
    tempfile.tempdir = str(scratch)
    (root/'clean_admission.json').write_text(json.dumps(admission,indent=2))
    source_layout = topological_patch_layout(clean,PATCH_COUNT)
    np.savez_compressed(root/'source_patch_layout.npz',**source_layout)
    records = []
    for slot in planned_slots(building_index):
        if selected and slot['variant_id'] not in selected:
            continue
        folder = root/'assets'/slot['variant_id']
        operations = recipe(slot,building_index,diagonal)
        mesh = clean
        remaining = np.ones(len(clean.faces),bool)
        support = np.zeros((len(clean.faces),6),np.float32)
        evidence = {}
        mappings_exact = True
        # Independent seeds per class, same class seed across all four targets.
        regions = {op['class']:MultiSurfaceRegion(clean,stable_seed(source.parent.name,op['class']),3)
                   for op in operations if 'fraction' in op}
        for op in operations:
            category, kind = op['class'], op['kind']
            index = ('G1','G2','G3','T1','T2','T3').index(category)
            weights = regions[category].weights(op['fraction']) if category in regions else np.ones(len(clean.faces))
            op['seed'] = stable_seed(source.parent.name,category)
            if category in regions:
                op['source_anchor_faces'] = regions[category].anchors
                evidence[f'{category}_region_ids'] = regions[category].region_ids
            support[:,index] = weights
            if kind == 'missing_faces':
                remaining &= weights == 0
                support[:,index] = ~remaining
            elif kind == 'correlated_noise':
                mesh = deform(mesh,weights>0,op['amplitude'],op['seed'])
            elif kind == 'bounded_smoothing':
                mesh = deform(mesh,weights>0,.08,op['seed'],smooth_steps=12,max_displacement=op['displacement_cap'])
            elif kind == 'textured_qem':
                mesh = textured_qem(mesh,op['retained'],calibrated=True)
                mappings_exact = False
            if kind in ('correlated_noise','bounded_smoothing'):
                offset = np.linalg.norm(mesh.vertices-clean.vertices,axis=1)
                evidence[f'{category}_vertex_displacement'] = offset
                support[:,index] = np.any(offset[clean.faces]>1e-9,axis=1)
        # Deleting last preserves source indexing during correlated deformation.
        if not remaining.all():
            if not mappings_exact:
                raise ValueError('Deletion+QEM requires an explicit correspondence implementation')
            mesh = replace(mesh,faces=mesh.faces[remaining],face_materials=mesh.face_materials[remaining])
        # Deleted surfaces cannot also count as an applied texture defect.
        support[:,3:] *= remaining[:,None]
        folder.mkdir(parents=True)
        path = export_geometry(clean,mesh,folder) if operations else stage_clean(source,folder)
        gltf = json.loads(path.read_text())
        texture_ops = [op for op in operations if op['class'].startswith('T')]
        # Resolve each material's image through texture.source, never assume image index==material index.
        for material, description in enumerate(gltf.get('materials',[])):
            texture = description.get('pbrMetallicRoughness',{}).get('baseColorTexture')
            if texture is None or not texture_ops:
                continue
            image_index = gltf['textures'][texture['index']]['source']
            image_record = gltf['images'][image_index]
            with Image.open(folder/image_record['uri']) as raw:
                image = raw.convert('RGBA')
            for number, op in enumerate(texture_ops):
                category, kind = op['class'], op['kind']
                if kind == 'downsample_encoding':
                    image = image.resize((max(1,round(image.width*op['ratio'])),max(1,round(image.height*op['ratio']))),Image.Resampling.LANCZOS)
                    # Alpha is retained for alpha-bearing textures; no arbitrary rejection or opacity conversion.
                    from io import BytesIO
                    stream = BytesIO()
                    alpha = image.getchannel('A')
                    image.convert('RGB').save(stream,format='JPEG',quality=op['jpeg_quality'],subsampling=2)
                    stream.seek(0)
                    image = Image.open(stream).convert('RGBA')
                    image.putalpha(alpha)
                    mask = np.ones((image.height,image.width),bool)
                else:
                    weights = support[:,('G1','G2','G3','T1','T2','T3').index(category)] * remaining
                    mask = surface_mask(clean,weights,material,image.size)
                    strength = op.get('strength',0)
                    before = np.asarray(image).copy()
                    if kind == 'projection_ghost':
                        valid_domain = surface_mask(clean,remaining.astype(float),material,image.size)
                        image = island_projection_ghost(image,mask,valid_domain,op['shift_relative'])
                    else:
                        effect = {'missing_texture':'missing','radiometric':'seam'}[kind]
                        image = local_texture(image,mask,effect,strength)
                    # Actual pixel changes differ from intended intervention support.
                    evidence[f'pixel_changed_op{number}_material{material}'] = np.any(
                        np.asarray(image)[...,:3] != before[...,:3], axis=2)
                evidence[f'pixel_support_op{number}_material{material}'] = mask
            target = folder/'textures'/f'processed_{material:04d}.png'
            image.save(target)
            image_record['uri'] = str(target.relative_to(folder))
        path.write_text(json.dumps(gltf,indent=2))
        np.savez_compressed(folder/'intervention_support.npz',source_face_attributes=support,
                            source_face_areas=area,source_face_retained=remaining,**evidence)
        current = GltfReader(path).load_mesh(include_texture=True)
        if (areas(current)>0).sum() < PATCH_COUNT:
            raise ValueError(f"{slot['variant_id']}: fewer than 16 effective faces; reject candidate")
        current_layout = topological_patch_layout(current,PATCH_COUNT)
        np.savez_compressed(folder/'current_patch_layout.npz',**current_layout)
        status = validate(path)
        record = {**slot,'schema_version':1,'protocol_version':VERSION,
                  'gltf':str(path.relative_to(root)), 'source_digest':source_digest,
                  'content_digest':status['asset_digest'],'decoded_digest':decoded_digest(current),
                  'operations':operations,'ratings':{},'formal_admitted':False,
                  'rating_status':'pending_independent_review','patch_count':PATCH_COUNT,
                  'patch_layout_version':'topological_v2_fixed16',
                  'current_source_face_mapping':'exact_surviving_face_order' if mappings_exact else 'unknown_after_qem',
                  'patch_quality_valid_mask':[[False]*3 for _ in range(PATCH_COUNT)],
                  'visible_attribute_valid_mask':[[False]*6 for _ in range(PATCH_COUNT)],
                  'support_note':'Source intervention evidence only; pixel masks may overlap reused UVs; visible labels remain unknown.',
                  'actual_source_area_fraction':(support*area[:,None]).sum(0).tolist()}
        record['actual_source_area_fraction'] = [x/float(area.sum()) for x in record['actual_source_area_fraction']]
        (folder/'metadata.json').write_text(json.dumps(record,indent=2))
        records.append(record)
        temporary = root/'candidate_manifest.tmp'
        temporary.write_text(json.dumps({'version':VERSION,'formal':False,'records':records},indent=2))
        temporary.replace(root/'candidate_manifest.json')
        print(slot['variant_id'],len(current.faces),'faces; independent grade pending',flush=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--clean-admission',type=Path,required=True)
    parser.add_argument('--building-index',type=int,required=True)
    parser.add_argument('--variants',nargs='+')
    args = parser.parse_args()
    if str(args.output).startswith('/Volumes/') and not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('请先插上移动硬盘 SANDISK-ELE')
    build(args.source,args.output,args.clean_admission,args.building_index,args.variants)
