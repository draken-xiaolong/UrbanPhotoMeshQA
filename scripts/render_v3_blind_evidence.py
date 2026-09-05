"""Serial offline candidate evidence with a grade/recipe-free public review queue."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from review_v3_visual_candidates import render
from urbanphotomeshqa.integrity import asset_digest, sha256_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if str(args.output).startswith('/Volumes/') and not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('请先插上移动硬盘 SANDISK-ELE')
    rows=json.loads((args.root/'candidate_manifest.json').read_text())['records']
    random.Random(2026).shuffle(rows)
    if args.limit is not None:
        rows=rows[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    public=[]; private=[]
    for row in rows:
        source=args.root/row['gltf']
        digest,_=asset_digest(source)
        if digest!=row['content_digest']:
            raise ValueError('Candidate content changed before review')
        identity=hashlib.sha256(('blind-review-v1:'+digest).encode()).hexdigest()[:20]
        folder=args.output/'public'/identity
        receipt=folder/'receipt.json'
        if receipt.exists():
            payload=json.loads(receipt.read_text())
            if payload['content_digest']!=digest or payload['size']!=args.size:
                raise ValueError('Review evidence version mismatch')
            if any(sha256_file(folder/name)!=value for name,value in payload['images'].items()):
                raise ValueError('Evidence changed; create a new review revision')
        else:
            render(source,folder,args.size)
            payload={'content_digest':digest,'size':args.size,
                     'images':{p.name:sha256_file(p) for p in sorted(folder.glob('view*.*'))}}
            receipt.write_text(json.dumps(payload,indent=2))
        # Reviewers receive only this list and neutral images, not the private map.
        public.append({'review_id':identity,'views':f'public/{identity}/views.jpg',
                       'detail_views':[f'public/{identity}/view{i}.png' for i in range(7)]})
        private.append({'review_id':identity,'content_digest':digest,'variant_id':row['variant_id'],
                        'gltf':row['gltf'],'target_scale':row['target_scale']})
        (args.output/'review_queue.json').write_text(json.dumps(public,indent=2))
        (args.output/'private_mapping.json').write_text(json.dumps(private,indent=2))
        print(identity,'evidence ready',flush=True)


if __name__=='__main__':
    main()
