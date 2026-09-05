"""Six-view visual audit and source-Face/Patch intervention labels for V3."""
import argparse
import json
import hashlib
from importlib.metadata import version
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.patches import topological_patch_layout
from urbanphotomeshqa.texture import render_textured_view


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--qem-only', action='store_true')
    p.add_argument('--finalize-only', action='store_true')
    p.add_argument('--support-only', action='store_true', help='Only compute patch support; skip legacy low-resolution renders')
    args = p.parse_args(); root = args.root
    records = json.loads((root/'manifest.json').read_text())['records']
    clean = GltfReader(records[0]['gltf_path']).load_mesh(include_texture=True)
    if args.finalize_only:
        layout = dict(np.load(root/'source_patch_layout.npz'))
    else:
        layout = topological_patch_layout(clean, 16)
        np.savez_compressed(root/'source_patch_layout.npz', **layout)
    directions = [(1,0,.2),(-1,0,.2),(0,1,.2),(0,-1,.2),(0,0,1),(1,1,.7)]
    previews = root/'previews'; previews.mkdir(exist_ok=True)
    results = []; baseline = None
    for record in records:
        if args.finalize_only:
            continue
        if args.qem_only and record['variant'] != 'clean' and 'qem' not in record['variant']:
            continue
        mesh = GltfReader(record['gltf_path']).load_mesh(include_texture=True)
        # Legacy renderer uses V-up; convert ONLY the audit copy from native glTF.
        uv = mesh.texcoords.copy(); uv[:,1] = 1-uv[:,1]
        mesh = replace(mesh, texcoords=uv)
        views = [] if args.support_only else [render_textured_view(mesh, direction=d, size=160) for d in directions]
        if baseline is None:
            baseline = views
        mae = [float(np.abs(a.astype(float)-b.astype(float)).mean()/255) for a,b in zip(views,baseline)]
        sheet = Image.new('RGB',(960,190),'white')
        ImageDraw.Draw(sheet).text((3,3),record['variant'],fill='black')
        for i,view in enumerate(views): sheet.paste(Image.fromarray(view),(i*160,30))
        if not args.support_only:sheet.save(previews/(record['variant']+'.png'))
        support_path = root/'assets'/record['variant']/'face_support.npz'
        with np.load(support_path) as support:
            weights = support['source_face_areas']; labels = layout['face_patch']
            patch = {}
            for kind in ('geometry','texture'):
                numerator = np.bincount(labels, weights=weights*support[kind], minlength=16)
                denominator = np.bincount(labels, weights=weights, minlength=16)
                patch[kind+'_intervention_fraction'] = (numerator/np.maximum(denominator,1e-12)).tolist()
        result = {'variant':record['variant'], 'six_view_rgb_mae':mae,
                  'visible_change_detected':max(mae)>1e-6 if mae else None, **patch}
        results.append(result)
        print(f"{len(results)}/50 visual audit {record['variant']}",flush=True)
    if args.qem_only or args.finalize_only:
        previous = json.loads((root/'visual_patch_audit.json').read_text())['records']
        changed = {r['variant'] for r in results}
        results += [r for r in previous if r['variant'] not in changed]
    (root/'visual_patch_audit.json').write_text(json.dumps({
        'note':'Visibility diagnostics and intervention support, not MOS or perceptual Patch truth',
        'views': directions,'records':results},indent=2))
    audit = json.loads((root/'audit.json').read_text())
    audit['six_view_audit_count'] = 0 if args.support_only else len(results)
    audit['no_visible_change'] = [r['variant'] for r in results if r['variant'] != 'clean' and r['visible_change_detected'] is False]
    audit['visual_review'] = 'support_only; visual_review_pending' if args.support_only else 'six_view_diagnostics_complete; human_scales_pending'
    audit['dependencies'] = {name: version(name) for name in ('pymeshlab','DracoPy','numpy','Pillow','scipy')}
    code_root = Path(__file__).resolve().parents[1]
    audit['code_sha256'] = {str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (code_root/'scripts/build_v3_process_pilot.py', code_root/'src/urbanphotomeshqa/process_degradations.py')}
    (root/'audit.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__': main()
