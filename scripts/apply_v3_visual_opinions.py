"""Publish explicit reviewed AI opinions, preserving all human scores and old AI revisions."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--asset-id',required=True)
p.add_argument('--revision',default='process_v2');p.add_argument('--config',type=Path,default=Path('configs/v3_visual_opinions_20260906.json'));a=p.parse_args()
config_path=a.config
config=json.loads(config_path.read_text());review=config['buildings'][a.asset_id]
root=Path('/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/HK3D-Individualised-V3/assets')/a.asset_id/a.revision
records=json.loads((root/'manifest.json').read_text())['records']
opinions={'clean':review['clean'],**review['combined']}
for category,values in review['single'].items():
    assert len(values)==4
    opinions.update({f'{category}_level{i+1}':value for i,value in enumerate(values)})
assert set(opinions)=={r['variant'] for r in records}
scores={};now=datetime.now(timezone.utc).isoformat()
for r in records:
    scale,reason=opinions[r['variant']];assert type(scale) is int and 1<=scale<=5
    evidence=root/'visual_review_v2'/r['variant']/'views.jpg'
    assert evidence.exists(),evidence
    scores[r['variant']]={'scale':scale,'reason':reason,'digest':r['content_digest'],
        'method':config['method'],'confidence':'provisional_pending_human','updated':now,
        'evidence':str(evidence),'evidence_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest(),
        'note':'AI视觉建议，非用户人工评分，非MOS。七视图与代表近景有限，需在交互网页复核。',
        'review_flags':['generation_realism_review'] if 'smooth' in r['variant'] else []}
    duplicates=[other['variant'] for other in records if other['variant']!=r['variant'] and other['content_digest']==r['content_digest']]
    if duplicates:
        scores[r['variant']]['review_flags'].append('duplicate_decoded_asset')
        scores[r['variant']]['reason']+=' 注意：与'+', '.join(duplicates)+'解码内容重复，不能计作独立正式训练样本。'
destination=root/'machine_scores.json'
if destination.exists():
    backup=root/'machine_score_history';backup.mkdir(exist_ok=True)
    shutil.copy2(destination,backup/('machine_scores_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'))
payload={'version':2,'method':config['method'],'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'scores':scores}
temporary=root/'machine_scores.pending.json';temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2));temporary.replace(destination)
print(a.asset_id,len(scores),'AI visual scores published; human database untouched')
