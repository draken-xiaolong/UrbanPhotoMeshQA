"""Bounded Train-only discovery views. Shortlists are NOT quality admissions."""
import argparse
import json
import random
import time
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
    p.add_argument('--resume', action='store_true')
    p.add_argument('--min-faces', type=int, default=16)
    p.add_argument('--exclude-plan', type=Path, nargs='*', default=[])
    args = p.parse_args()
    if not 1 <= args.count <= 12:
        raise ValueError('Discovery batch must be bounded to 1..12 buildings')
    if str(args.output).startswith('/Volumes/') and not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('请先插上移动硬盘 SANDISK-ELE')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Do not overwrite previous screening')
    for path in args.exclude_plan:
        old = json.loads(path.read_text())
        args.exclude.extend(old.get('excluded_ids', []))
        args.exclude.extend(r['asset_id'] for r in old['records'])
    args.exclude = sorted(set(args.exclude))
    records = json.loads(args.audit.read_text())['records']
    pool = sorted((r for r in records if r['split']=='train' and r['status']=='qualified'
                   and r['asset_id'] not in args.exclude
                   and r['face_count'] >= args.min_faces
                   and (args.source_root / r['source_gltf']).is_file()), key=lambda r:r['asset_id'])
    random.Random(2026).shuffle(pool)
    chosen = pool[:args.count]
    if not chosen:
        raise ValueError('No eligible unreviewed Train sources on host')
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {'seed':2026,'formal_admitted':False,'selection':'Train qualified files, shuffled; no quality score used',
            'excluded_ids':args.exclude,'records':chosen}
    plan_path = args.output/'screening_plan.json'
    if args.resume:
        plan = json.loads(plan_path.read_text())
        chosen = plan['records']
    else:
        plan['minimum_faces'] = args.min_faces
        plan_path.write_text(json.dumps(plan,indent=2))
    results = []
    for row in chosen:
        source = args.source_root / row['source_gltf']
        folder = args.output / row['asset_id']
        started = time.monotonic()
        try:
            digest,_ = asset_digest(source)
            receipt_path = folder/'receipt.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if (receipt['content_digest'] != digest or receipt['size'] != 512
                    or receipt['directions'] != [5,6,4]
                    or set(receipt['images']) != {'view0.png','view1.png','view2.png','views.jpg'}
                    or any(sha256_file(folder/name) != value for name,value in receipt['images'].items())):
                    raise ValueError('Existing evidence mismatch; retain and use a new revision')
                results.append({'asset_id':row['asset_id'],'status':'reused'})
                continue
            if folder.exists():
                raise ValueError('Partial evidence retained; retry this asset in a new revision')
            stats = render(source,folder,512,[DIRECTIONS[i] for i in (5,6,4)],material_aware=True)
            receipt = {'content_digest':digest,'size':512,'directions':[5,6,4],
                   'quality_rating':None,'formal_admitted':False,'statistics':stats,
                   'elapsed_seconds':time.monotonic()-started,
                   'images':{f.name:sha256_file(f) for f in folder.glob('view*.*')}}
            receipt_path.write_text(json.dumps(receipt,indent=2))
            results.append({'asset_id':row['asset_id'],'status':'completed','elapsed_seconds':receipt['elapsed_seconds']})
            print(row['asset_id'],'discovery evidence complete; not quality-admitted',flush=True)
        except Exception as error:
            results.append({'asset_id':row['asset_id'],'status':'failed','error':str(error)})
            print(row['asset_id'],'FAILED',str(error),flush=True)
        finally:
            (args.output/'run_status.json').write_text(json.dumps({'formal_admitted':0,'results':results},indent=2))


if __name__ == '__main__':
    main()
