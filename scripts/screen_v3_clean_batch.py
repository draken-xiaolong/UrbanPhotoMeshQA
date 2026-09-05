"""Bounded Train-only discovery views. Shortlists are NOT quality admissions."""
import argparse
import json
import random
from pathlib import Path
from review_v3_visual_candidates import DIRECTIONS, render
from urbanphotomeshqa.integrity import asset_digest, sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclude', nargs='*', default=[])
    p.add_argument('--count', type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.count <= 12:
        raise ValueError('Discovery batch must be bounded to 1..12 buildings')
    if str(args.output).startswith('/Volumes/') and not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('请先插上移动硬盘 SANDISK-ELE')
    if args.output.exists():
        raise FileExistsError('Do not overwrite previous screening')
    records = json.loads(args.audit.read_text())['records']
    pool = sorted((r for r in records if r['split']=='train' and r['status']=='qualified'
                   and r['asset_id'] not in args.exclude
                   and (args.source_root / r['source_gltf']).is_file()), key=lambda r:r['asset_id'])
    random.Random(2026).shuffle(pool)
    chosen = pool[:args.count]
    if not chosen:
        raise ValueError('No eligible unreviewed Train sources on host')
    args.output.mkdir(parents=True)
    plan = {'seed':2026,'formal_admitted':False,'selection':'Train qualified files, shuffled; no quality score used',
            'excluded_ids':args.exclude,'records':chosen}
    (args.output/'screening_plan.json').write_text(json.dumps(plan,indent=2))
    for row in chosen:
        source = args.source_root / row['source_gltf']
        folder = args.output / row['asset_id']
        digest,_ = asset_digest(source)
        stats = render(source,folder,512,[DIRECTIONS[i] for i in (5,6,4)],material_aware=True)
        receipt = {'content_digest':digest,'size':512,'directions':[5,6,4],
                   'quality_rating':None,'formal_admitted':False,'statistics':stats,
                   'images':{f.name:sha256_file(f) for f in folder.glob('view*.*')}}
        (folder/'receipt.json').write_text(json.dumps(receipt,indent=2))
        print(row['asset_id'],'discovery evidence complete; not quality-admitted',flush=True)


if __name__ == '__main__':
    main()
