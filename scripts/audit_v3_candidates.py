"""One-time lightweight received-package audit; no rendering or quality grading."""
import argparse
import json
from pathlib import Path

from urbanphotomeshqa.integrity import asset_digest, sha256_file
from urbanphotomeshqa.v3_attribute_labels import compile_labels
from urbanphotomeshqa.v3_protocol import VERSION, PATCH_COUNT


def audit(root):
    root = root.resolve()
    manifest = root/'candidate_manifest.json'
    payload = json.loads(manifest.read_text())
    if payload.get('version') != VERSION or payload.get('formal') is not False:
        raise ValueError('Expected non-formal V3 candidates')
    rows = payload['records']
    if not rows:
        raise ValueError('Empty candidate manifest')
    names, digests, decoded = set(), set(), set()
    verified = []
    for row in rows:
        path = (root/row['gltf']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Asset escapes candidate root')
        digest, dependencies = asset_digest(path)
        if digest != row['content_digest']:
            raise ValueError(f"Content mismatch: {row['variant_id']}")
        if row['variant_id'] in names or digest in digests or row['decoded_digest'] in decoded:
            raise ValueError('Duplicate candidate ID/content/decoded content')
        names.add(row['variant_id']); digests.add(digest); decoded.add(row['decoded_digest'])
        if row.get('patch_count') != PATCH_COUNT:
            raise ValueError('Invalid Patch count')
        layout_digest = sha256_file(path.parent/'current_patch_layout.npz')
        if layout_digest != row.get('patch_layout_digest'):
            raise ValueError('Patch layout changed or missing layout binding')
        label_path = (path.parent/row['visible_attribute_labels']).resolve()
        if not label_path.is_relative_to(path.parent):
            raise ValueError('Labels escape asset package')
        compiled = compile_labels(json.loads(label_path.read_text()), digest, layout_digest)
        verified.append(dict(variant_id=row['variant_id'], content_digest=digest,
            dependencies=len(dependencies), bytes=sum(x['bytes'] for x in dependencies),
            patch_layout_digest=layout_digest, labels_sha256=sha256_file(label_path),
            visible_building_labels=int(compiled['building_valid'].sum()),
            visible_patch_labels=int(compiled['patches_valid'].sum())))
    return dict(manifest_sha256=sha256_file(manifest), count=len(verified),
                audit_type='received_package_hash_and_label_binding', passed=True,
                formal_admitted=False, records=verified,
                limitations='Does not recompute decoded geometry digest, perceptual grades, or physical validity')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if str(args.output).startswith('/Volumes/') and not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('请先插上移动硬盘 SANDISK-ELE')
    if args.output.exists():
        raise FileExistsError('Audit exists; reuse it for unchanged content or select a new revision')
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"{report['count']} packages verified; formal admission remains false")
