#!/usr/bin/env python3
"""Build one surface-aware 50-version pilot; never overwrite an existing revision."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import numpy as np
from PIL import Image
from build_iteration3_pilot import export_geometry, stable_seed, validate
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.process_degradations import (
    SurfaceRegion, areas, deform, textured_qem, draco_positions, surface_mask, local_texture,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asset-root', type=Path, required=True)
    p.add_argument('--revision', default='process_v2')
    p.add_argument('--repair-qem', action='store_true')
    p.add_argument('--resume', action='store_true', help='Reuse completed per-variant metadata after an interrupted pilot')
    p.add_argument('--calibration', type=Path, help='Explicit new-revision parameter configuration')
    args = p.parse_args()
    root = args.asset_root / args.revision
    root.mkdir(exist_ok=args.repair_qem or args.resume)
    source = args.asset_root / 'clean' / 'clean_scale5.gltf'
    clean = GltfReader(source).load_mesh(include_texture=True)
    seed = stable_seed(args.asset_root.name, 'surface-process')
    region = SurfaceRegion(clean, seed)
    records = []
    if args.repair_qem:
        records = json.loads((root/'manifest.json').read_text())['records']
        records = [r for r in records if 'qem' not in r['variant']]
    # Four strength values, deliberately independent of the review scale.
    parameters = {
        'geometry_missing': [.005, .02, .06, .15],
        'geometry_artifacts': [.015, .045, .12, .30],
        'mesh_simplification_qem': [.85, .60, .35, .15],
        'geometry_smoothing': [3, 7, 15, 30],
        'position_quantization': [16, 13, 11, 9],
        'uv_quantization': [14, 12, 10, 8],
        'texture_blur_resolution_loss': [.75, .50, .25, .125],
        'texture_compression': [85, 60, 35, 12],
        'texture_missing_occlusion': [.015, .05, .15, .30],
        'texture_misalignment_uv': [1., 3., 7., 15.],
        'texture_seam_radiometric': [.08, .2, .45, .85],
    }
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    if calibration:
        parameters = calibration['parameters']
        if calibration.get('scratch_root'):
            scratch=Path(calibration['scratch_root'])
            if not Path('/Volumes/SANDISK-ELE').is_mount():
                raise ValueError('请先插上移动硬盘 SANDISK-ELE')
            scratch.mkdir(parents=True,exist_ok=True)
            tempfile.tempdir=str(scratch)
        if args.revision == 'process_v2':
            raise ValueError('Calibration requires a new revision, never overwrite process_v2')
        configuration_path = root/'generation_protocol.json'
        if configuration_path.exists() and json.loads(configuration_path.read_text()) != calibration:
            raise ValueError('Resume configuration mismatch; use a new revision')
        configuration_path.write_text(json.dumps(calibration, indent=2))
    local_kinds = ['geometry_missing','geometry_artifacts','geometry_smoothing',
                   'texture_missing_occlusion','texture_misalignment_uv','texture_seam_radiometric']
    regions = {kind:region for kind in local_kinds}
    anchors = {}
    if calibration:
        # Spatial farthest-point anchors, seeded per building. Same type keeps
        # one nested region across all levels, different types spread out.
        centers = clean.vertices[clean.faces].mean(axis=1)
        rng = np.random.default_rng(seed)
        valid = np.flatnonzero(region.area > 0)
        face_patch = None
        if calibration.get('distinct_patch_anchors'):
            from urbanphotomeshqa.patches import topological_patch_layout
            face_patch = topological_patch_layout(clean,16)['face_patch']
        chosen = [int(rng.choice(valid))]
        for _ in range(len(local_kinds)-1):
            distance = np.min(np.linalg.norm(centers[:,None]-centers[chosen],axis=2),axis=1)
            available = valid
            if face_patch is not None:
                available = valid[~np.isin(face_patch[valid],face_patch[chosen])]
                if not len(available):raise ValueError('Not enough distinct patches for local anchors')
            chosen.append(int(available[np.argmax(distance[available])]))
        for kind, anchor in zip(rng.permutation(local_kinds),chosen):
            anchors[str(kind)] = anchor
            regions[str(kind)] = SurfaceRegion(clean, stable_seed(args.asset_root.name,str(kind)),anchor)
    subtypes = {'geometry_artifacts': 'spatially_correlated_surface_noise',
                'texture_blur_resolution_loss': 'native_texture_downsample',
                'texture_missing_occlusion': 'missing_texture_fill',
                'texture_misalignment_uv': 'surface_masked_projection_ghost',
                'texture_seam_radiometric': 'surface_region_exposure_mismatch'}

    def generate(name, operations):
        if args.repair_qem and 'qem' not in name:
            return
        folder = root / 'assets' / name
        if args.resume and (folder/'metadata.json').exists():
            completed=json.loads((folder/'metadata.json').read_text())
            if validate(Path(completed['gltf_path']))['asset_digest']!=completed['asset_digest']:
                raise ValueError(f'Resume content changed: {name}')
            records.append(completed)
            print(f'{len(records)}/50 reuse {name}',flush=True)
            return
        if folder.exists():
            archived = root/('interrupted_variants' if args.resume else 'superseded_qem')
            archived.mkdir(exist_ok=True)
            shutil.move(str(folder), str(archived/name))
        folder.mkdir(parents=True)
        mesh = clean
        geo_affected = np.zeros(len(clean.faces), bool)
        tex_affected = np.zeros(len(clean.faces), bool)
        texture_ops = []
        recorded = []
        draco_bytes = None
        for kind, level in operations:
            region = regions.get(kind, next(iter(regions.values())))
            value = parameters[kind][level-1]
            recorded.append({'category': kind, 'subtype': subtypes.get(kind, kind),
                             'level': level, 'value': value, 'anchor_face':anchors.get(kind)})
            if kind == 'geometry_missing':
                selected = region.mask(value)
                geo_affected |= selected
                mesh = replace(mesh, faces=mesh.faces[~selected], face_materials=mesh.face_materials[~selected])
            elif kind == 'geometry_artifacts':
                selected = region.mask(.25)
                mesh = deform(mesh, selected, value, seed)
                geo_affected |= selected
            elif kind == 'geometry_smoothing':
                selected = region.mask(.35)
                if calibration:
                    mesh = deform(mesh, selected, .08, seed, smooth_steps=12, max_displacement=value)
                else:
                    mesh = deform(mesh, selected, .25, seed, smooth_steps=value)
                geo_affected |= selected
            elif kind == 'mesh_simplification_qem':
                mesh = textured_qem(mesh, value, calibrated=bool(calibration))
                geo_affected[:] = True
            elif kind == 'position_quantization':
                mesh, draco_bytes = draco_positions(mesh, value)
                geo_affected[:] = True
            elif kind == 'uv_quantization':
                # Draco quantizes UV as a 2D point field embedded in XYZ; explicit
                # standalone attribute coding, not claimed as a full glTF codec.
                import DracoPy
                valid_uv=np.isfinite(mesh.texcoords).all(axis=1)
                points = np.column_stack([mesh.texcoords[valid_uv], np.zeros(int(valid_uv.sum()))])
                payload = DracoPy.encode(points, quantization_bits=value, preserve_order=True)
                uv = DracoPy.decode(payload).points[:, :2]
                if len(uv) != int(valid_uv.sum()):
                    raise ValueError('UV codec changed vertex indexing')
                restored_uv=mesh.texcoords.copy();restored_uv[valid_uv]=uv
                mesh = replace(mesh, texcoords=restored_uv)
                (folder / 'uv_attribute_payload.drc').write_bytes(payload)
                tex_affected[:] = True
            else:
                coverage = (calibration or {}).get('local_texture_coverage',[.30]*4)[level-1]
                texture_ops.append((kind, value, coverage))
                if kind == 'texture_missing_occlusion':
                    tex_affected |= region.mask(value)
                elif kind in ('texture_misalignment_uv', 'texture_seam_radiometric'):
                    tex_affected |= region.mask(coverage)
                else:
                    tex_affected[:] = True
        path = export_geometry(clean, mesh, folder, 'model.gltf') if operations else source
        if draco_bytes is not None:
            (folder / 'position_payload.drc').write_bytes(draco_bytes)
        if texture_ops:
            gltf = json.loads(path.read_text())
            # Exporter writes one base-color texture per material in material order.
            for material, image_record in enumerate(gltf['images']):
                if not clean.metadata['material_texture_paths'][material]:
                    continue  # No texture exists to degrade on constant-color material.
                texture_path = folder / image_record['uri']
                image = Image.open(texture_path).convert('RGBA')
                for kind, value, coverage in texture_ops:
                    region = regions.get(kind, next(iter(regions.values())))
                    if kind == 'texture_blur_resolution_loss':
                        image = image.resize((max(1, round(image.width*value)), max(1, round(image.height*value))), Image.Resampling.LANCZOS)
                    elif kind == 'texture_compression':
                        # Real encoded JPEG payload retained; alpha-bearing assets rejected.
                        if np.asarray(image.getchannel('A')).min() != 255:
                            raise ValueError('JPEG pilot requires opaque textures')
                        target = folder / 'textures' / f'material_{material:02d}_q{value}.jpg'
                        image.convert('RGB').save(target, quality=value, subsampling=2)
                        image = Image.open(target).convert('RGBA')
                    else:
                        fraction = value if kind == 'texture_missing_occlusion' else coverage
                        selected = region.weights(fraction) if (calibration or {}).get('fractional_texture_budget') else region.mask(fraction)
                        mask = surface_mask(clean, selected, material, image.size)
                        effect = {'texture_missing_occlusion': 'missing', 'texture_misalignment_uv': 'misalignment', 'texture_seam_radiometric': 'seam'}[kind]
                        image = local_texture(image, mask, effect, value)
                if texture_ops[-1][0] == 'texture_compression':
                    image_record['uri'] = str(target.relative_to(folder))
                else:
                    target = folder / 'textures' / f'material_{material:02d}_processed.png'
                    image.save(target)
                    image_record['uri'] = str(target.relative_to(folder))
            path.write_text(json.dumps(gltf, indent=2))
        region = next(iter(regions.values()))
        if calibration and any(k in ('geometry_artifacts','geometry_smoothing') for k,_ in operations):
            if len(mesh.vertices)==len(clean.vertices):
                changed=np.linalg.norm(mesh.vertices-clean.vertices,axis=1)>1e-9
                geo_affected |= np.any(changed[clean.faces],axis=1)
        np.savez_compressed(folder / 'face_support.npz',
                            geometry=geo_affected, texture=tex_affected,
                            source_face_areas=region.area)
        # Canonical decoded content digest ignores filenames and packaging.
        decoded = GltfReader(path).load_mesh(include_texture=True)
        h = hashlib.sha256()
        for a in (decoded.vertices, decoded.faces, decoded.texcoords, decoded.normals):
            h.update(np.ascontiguousarray(a).tobytes())
        for tp in decoded.metadata['material_texture_paths']:
            if tp is None:
                h.update(b'constant-color-material')
                continue
            with Image.open(tp) as im:
                h.update(str(im.size).encode()); h.update(im.convert('RGBA').tobytes())
        status = validate(path)
        t = decoded.vertices[decoded.faces]
        zero_area = int(np.count_nonzero(np.linalg.norm(np.cross(t[:,1]-t[:,0], t[:,2]-t[:,0]), axis=1)<1e-12))
        record = {'asset_id': args.asset_root.name, 'variant': name, 'gltf_path': str(path),
                  'human_rating_initial': None if calibration else (5 if not operations else 3),
                  'rating_status': 'unreviewed_placeholder' if operations else 'original_requires_review',
                  'generation_protocol': calibration['protocol'] if calibration else 'process_v2',
                  'operations': recorded, 'seed': seed, 'content_digest': h.hexdigest(),
                  'zero_area_faces': zero_area,
                  'geometry_support_area_fraction': float(region.area[geo_affected].sum()/region.area.sum()),
                  'texture_support_area_fraction': float(region.area[tex_affected].sum()/region.area.sum()),
                  'support_note': 'Source-face intervention support; not perceptual quality ground truth',
                  **status}
        (folder / 'metadata.json').write_text(json.dumps(record, indent=2))
        records.append(record)
        print(f'{len(records)}/50 {name}', flush=True)

    generate('clean', [])
    for kind in parameters:
        if kind == 'uv_quantization':
            continue
        for level in range(1, 5):
            generate(f'{kind}_level{level}', [(kind, level)])
    combinations = [
        ('qem_downsample', [('mesh_simplification_qem',2), ('texture_blur_resolution_loss',2)]),
        ('position_jpeg', [('position_quantization',3), ('texture_compression',3)]),
        ('qem_uv_jpeg', [('mesh_simplification_qem',3), ('uv_quantization',3), ('texture_compression',3)]),
        ('hole_missing', [('geometry_missing',3), ('texture_missing_occlusion',3)]),
        ('noise_ghost', [('geometry_artifacts',3), ('texture_misalignment_uv',3)]),
        ('smooth_downsample', [('geometry_smoothing',3), ('texture_blur_resolution_loss',3)]),
        ('hole_ghost', [('geometry_missing',3), ('texture_misalignment_uv',3)]),
        ('downsample_seam', [('texture_blur_resolution_loss',2), ('texture_seam_radiometric',3)]),
        ('qem_jpeg', [('mesh_simplification_qem',2), ('texture_compression',3)]),
    ]
    for name, ops in combinations:
        generate('combined_'+name, ops)
    records.sort(key=lambda r: (r['variant'] != 'clean', r['variant']))
    hashes = [r['content_digest'] for r in records]
    audit = {'versions': len(records), 'unique_decoded_assets': len(set(hashes)),
             'all_parseable': True, 'visual_review': 'pending',
             'zero_area_faces': {r['variant']: r['zero_area_faces'] for r in records if r['zero_area_faces']}}
    (root / 'manifest.json').write_text(json.dumps({'seed':2026, 'records':records}, indent=2))
    (root / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
