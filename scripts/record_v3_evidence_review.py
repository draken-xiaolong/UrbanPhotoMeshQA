"""Record an explicitly supplied visual opinion, binding assets and all evidence."""
import argparse
import json
from pathlib import Path
from urbanphotomeshqa.integrity import asset_digest, sha256_file


def record(source, evidence, audit, scale, reason):
    if scale not in range(1,6) or not reason.strip():
        raise ValueError('Explicit grade and visual rationale required')
    digest, dependencies = asset_digest(source)
    receipt = json.loads((evidence/'receipt.json').read_text())
    if receipt['content_digest'] != digest:
        raise ValueError('Evidence does not match the current asset')
    expected = {f'view{i}.png' for i in range(7)} | {'views.jpg'}
    if set(receipt['images']) != expected:
        raise ValueError('Complete seven-view evidence required')
    for name, value in receipt['images'].items():
        if sha256_file(evidence/name) != value:
            raise ValueError('Evidence image changed')
    rows = json.loads(audit.read_text())['records']
    matches = [r for r in rows if r['asset_id']==source.parent.name and r['asset_digest']==digest
               and r['status']=='qualified']
    if len(matches) != 1:
        raise ValueError('No unique matching technical source audit')
    return {'asset_id':source.parent.name,'source_gltf':matches[0]['source_gltf'],
            'technical_valid':True,'technical_evidence':'source hash matches qualified historical audit',
            'source_dependencies':dependencies,'formal_split':None,'protocol_frozen':False,
            'role':'development_calibration' if scale==5 else 'not_admitted_as_scale5_anchor',
            'ratings':{'machine':{'scale':scale,'content_digest':digest,'reason':reason,
            'uncertain':False,'reviewer':'Codex_AI','human_mos':False,
            'protocol_version':'v3_visual_evidence_v2_development',
            'evidence':[{'path':str(evidence/name),'sha256':value} for name,value in receipt['images'].items()],
            'limitation':'Offline seven fixed views; grade is an explicit AI opinion, not automatic score or human MOS'}}}


def main():
    p=argparse.ArgumentParser()
    for name in ('source','evidence','audit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scale',type=int,required=True)
    p.add_argument('--reason',required=True)
    args=p.parse_args()
    if str(args.output).startswith('/Volumes/') and not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('请先插上移动硬盘 SANDISK-ELE')
    result=record(args.source,args.evidence,args.audit,args.scale,args.reason)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(result['asset_id'],result['role'])


if __name__=='__main__':
    main()
