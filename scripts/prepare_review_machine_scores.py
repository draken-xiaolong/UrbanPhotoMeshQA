"""Transparent, provisional visual-diagnostic scores; never human MOS."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    destination = root/'machine_scores.json'
    if destination.exists(): raise SystemExit('Machine scores already exist; use a new version to revise them.')
    records = json.loads((root/'manifest.json').read_text())['records']
    audit = {r['variant']:r for r in json.loads((root/'visual_patch_audit.json').read_text())['records']}
    scores = {}
    for r in records:
        error = max(audit[r['variant']]['six_view_rgb_mae'])
        # Diagnostic thresholds are provisional, not learned or human-calibrated.
        scale = 5-sum(error >= t for t in (.002,.015,.04,.08))
        if r['variant']=='clean': scale=5
        scores[r['variant']] = {'scale':scale,'digest':r['content_digest'],
            'method':'six_view_diagnostic_v1','confidence':'uncalibrated',
            'reason':f'六视角最大RGB平均差 {error:.5f}；临时阈值 0.002 / 0.015 / 0.04 / 0.08',
            'note':'参考辅助的启发式机器建议分，未经人工标定；不是MOS或模型预测。',
            'updated':datetime.now(timezone.utc).isoformat()}
    destination.write_text(json.dumps({'version':1,'scores':scores},ensure_ascii=False,indent=2))
    print(f'Prepared {len(scores)} machine scores: {destination}')


if __name__=='__main__':main()
