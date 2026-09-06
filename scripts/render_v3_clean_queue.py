"""Bounded, resumable seven-view Clean evidence; no implicit quality grading."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

from urbanphotomeshqa.integrity import sha256_file


def main():
    p = argparse.ArgumentParser()
    for name in ('audit', 'source-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, choices=range(1, 5), default=2)
    p.add_argument('--timeout', type=int, default=180)
    args = p.parse_args()
    if str(args.output).startswith('/Volumes/') and not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('External disk is not mounted')
    rows = json.loads(args.audit.read_text())['records']
    if len(rows) > 240:
        raise ValueError('Queue exceeds the 240 building bound')
    args.output.mkdir(parents=True, exist_ok=True)
    expected = {f'view{i}.png' for i in range(7)} | {'views.jpg'}

    def run(row):
        started = time.monotonic()
        folder = args.output/row['asset_id']
        try:
            receipt = folder/'receipt.json'
            if receipt.exists():
                evidence = json.loads(receipt.read_text())
                if (evidence['content_digest'] != row['asset_digest'] or evidence['size'] != 1024
                    or set(evidence['images']) != expected
                    or any(sha256_file(folder/n) != h for n, h in evidence['images'].items())):
                    raise ValueError('Existing evidence mismatch; retained without overwrite')
                status = 'reused'
            else:
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('render_v3_clean_evidence.py')),
                    '--source', str(args.source_root/row['source_gltf']), '--output', str(folder), '--size', '1024'],
                    capture_output=True, text=True, timeout=args.timeout)
                if result.returncode:
                    raise RuntimeError(result.stderr[-2000:])
                status = 'completed'
            return {'asset_id': row['asset_id'], 'status': status, 'elapsed_seconds': time.monotonic()-started}
        except Exception as error:
            return {'asset_id': row['asset_id'], 'status': 'failed', 'error': str(error)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, row) for row in rows]
        with (args.output/'run_events.jsonl').open('a') as stream:
            for future in as_completed(futures):
                result = future.result()
                stream.write(json.dumps(result)+'\n'); stream.flush()
                print(result['asset_id'], result['status'], flush=True)


if __name__ == '__main__':
    main()
