"""Bind already locked anonymous AI opinions to evidence; never admit candidates."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(evidence, scores):
    opinions = json.loads(scores.read_text())
    if opinions.get('status') != 'independent_opinions_locked_before_mapping':
        raise ValueError('Save independent opinions before opening the private map')
    rows = opinions['ratings']
    ids = [row['review_id'] for row in rows]
    queue = json.loads((evidence / 'review_queue.json').read_text())
    if len(set(ids)) != len(ids) or set(ids) != {r['review_id'] for r in queue}:
        raise ValueError('Scores must cover the complete queue exactly once')
    private = json.loads((evidence / 'private_mapping.json').read_text())
    mapping = {row['review_id']: row for row in private}
    if len(mapping) != len(private) or set(mapping) != set(ids):
        raise ValueError('Private map does not match the complete queue')
    result = []
    for row in rows:
        if type(row['scale']) is not int or row['scale'] not in range(1, 6) or not row['reason'].strip():
            raise ValueError('Invalid independent grade or empty rationale')
        folder = evidence / 'public' / row['review_id']
        receipt_path = folder / 'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        expected_images = {f'view{i}.png' for i in range(7)} | {'views.jpg'}
        if set(receipt['images']) != expected_images:
            raise ValueError('Incomplete image evidence')
        for name, expected in receipt['images'].items():
            if digest(folder / name) != expected:
                raise ValueError('Image evidence changed')
        candidate = mapping[row['review_id']]
        if receipt['content_digest'] != candidate['content_digest']:
            raise ValueError('Candidate and evidence digests disagree')
        result.append({**candidate, 'machine_scale': row['scale'], 'reason': row['reason'],
                       'rating_source': 'AI_visual_opinion_not_human_MOS',
                       'review_scope': opinions.get('scope'),
                       'review_limitations': opinions.get('limitations'),
                       'evidence_receipt_sha256': digest(receipt_path),
                       'matches_target': row['scale'] == candidate['target_scale'],
                       'formal_admitted': False,
                       'remaining_gates': ['physical_plausibility', 'visible_patch_labels', 'formal_protocol']})
    result.sort(key=lambda row: row['variant_id'])
    return {'scores_sha256': digest(scores), 'scores_path': str(scores),
            'note': 'Self-reported blind procedure, not a cryptographic proof of reviewer isolation. Matching a target is not formal admission.',
            'count': len(result), 'target_matches': sum(r['matches_target'] for r in result),
            'grade_counts': dict(Counter(r['machine_scale'] for r in result)),
            'records': result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if str(args.output).startswith('/Volumes/') and not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('请先插上移动硬盘 SANDISK-ELE')
    report = summarize(args.evidence, args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print('count', report['count'], 'target_matches', report['target_matches'])
    for row in report['records']:
        print(row['variant_id'], 'target', row['target_scale'], 'observed', row['machine_scale'])


if __name__ == '__main__':
    main()
