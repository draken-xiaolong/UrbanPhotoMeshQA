"""Finite material-aware Clean screening on the GPU host; no automatic grades."""
import argparse
import json
from pathlib import Path
from review_v3_visual_candidates import render
from urbanphotomeshqa.integrity import asset_digest, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=1024)
    args = parser.parse_args()
    if str(args.output).startswith('/Volumes/') and not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('请先插上移动硬盘 SANDISK-ELE')
    if args.output.exists():
        raise FileExistsError('Use a new evidence revision; never overwrite past reviews')
    digest, _ = asset_digest(args.source)
    stats = render(args.source, args.output, args.size, material_aware=True)
    receipt = {'content_digest': digest, 'renderer': 'material_aware_v2', 'size': args.size,
               'source': str(args.source), 'statistics': stats, 'quality_rating': None,
               'images': {p.name: sha256_file(p) for p in sorted(args.output.glob('view*.*'))}}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(args.source.parent.name, 'evidence complete; not quality-admitted', flush=True)


if __name__ == '__main__':
    main()
