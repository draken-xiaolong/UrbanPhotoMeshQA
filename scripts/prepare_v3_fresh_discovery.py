"""Bounded technical screening of previously unused sources, never quality admission."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from select_iteration2_source_assets import inspect_asset
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.integrity import asset_digest


def main():
    p = argparse.ArgumentParser()
    for name in ('inventory', 'historical-audit', 'source-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--count', type=int, default=240)
    args = p.parse_args()
    if not 1 <= args.count <= 240:
        raise ValueError('Bounded to 240 source packages per revision')
    if str(args.output).startswith('/Volumes/') and not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('External disk is not mounted')
    old = json.loads(args.historical_audit.read_text())['records']
    excluded = {r['asset_id'] for r in old}
    tiles = {r['tile'] for r in old if r['split'] == 'train'}
    rows = json.loads(args.inventory.read_text())['records']
    rows = [r for r in rows if r['asset_id'] not in excluded and r['tile'] in tiles]
    rows.sort(key=lambda r: hashlib.sha256(('2026:fresh-discovery:'+r['asset_id']).encode()).hexdigest())
    selected = rows[:args.count]
    # A frozen source queue can be resumed; completed packages are not decoded again.
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output/'plan.json'
    if plan_path.exists():
        selected = json.loads(plan_path.read_text())['records']
    else:
        with plan_path.open('x') as f:
            json.dump({'records': selected, 'excluded_historical_ids': sorted(excluded),
                       'seed': 2026, 'formal_admitted': 0, 'purpose': 'Train discovery only'}, f, indent=2)
    records, failures = [], []
    for row in selected:
        receipt_path = args.output/(row['asset_id']+'.json')
        if receipt_path.exists():
            result = json.loads(receipt_path.read_text())
        else:
            source = args.source_root/row['source_gltf']
            try:
                result = inspect_asset(source, args.source_root)
                result.update(split='train', formal_admitted=False, quality_scale=None)
                if result['status'] == 'qualified' and result['face_count'] >= 256:
                    digest, deps = asset_digest(source)
                    mesh = GltfReader(source).load_mesh(include_texture=True)
                    if (not np.isfinite(mesh.vertices).all() or mesh.texcoords is None
                        or not np.isfinite(mesh.texcoords).all()):
                        raise ValueError('Non-finite geometry or missing/invalid UV')
                    result.update(asset_digest=digest, dependencies=deps,
                                  parsed_face_count=len(mesh.faces), technical_check='full_decode_finite_uv_v1')
                else:
                    result['status'] = 'not_selected_for_discovery'
            except Exception as error:
                result = {**row, 'status': 'failed', 'error': str(error)}
            with receipt_path.open('x') as f:
                json.dump(result, f, indent=2)
        (records if result['status'] == 'qualified' else failures).append(result)
        out = {'records': records, 'not_selected_or_failed': failures, 'formal_admitted': 0,
               'quality_scale': None, 'seed': 2026, 'scope': 'fresh Train candidate technical audit'}
        temp = args.output/'audit.pending.json'
        temp.write_text(json.dumps(out, indent=2))
        temp.replace(args.output/'audit.json')
        print(row['asset_id'], result['status'], flush=True)


if __name__ == '__main__':
    main()
