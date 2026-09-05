"""Recoverable, same-volume archival of explicitly listed V3 trial directories."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/HK3D-Individualised-V3')
NAMES = ('assets', '_review', 'audits', 'human_ratings', 'manifests', 'patch_targets', 'previews')


def archive(root=ROOT, execute=False):
    if not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('请先插上移动硬盘 SANDISK-ELE')
    if root.resolve() != ROOT or root.is_symlink():
        raise ValueError('Only the explicit V3 trial root is allowed')
    sources = [root / name for name in NAMES if (root / name).exists()]
    if any(p.is_symlink() or not p.is_dir() for p in sources):
        raise ValueError('Unexpected source type; archival refused')
    destination = root / '_archive' / datetime.now(timezone.utc).strftime('pre_6class33_%Y%m%dT%H%M%SZ')
    plan = {'schema_version': 1, 'mode': 'same_volume_rename_no_deletion',
            'root': str(root), 'destination': str(destination),
            'entries': [{'source': str(p), 'destination': str(destination / p.name),
                         'inode': p.stat().st_ino, 'status': 'pending'} for p in sources],
            'note': 'Old absolute paths resolve via this mapping; archive contents are unchanged.'}
    if not execute:
        return plan
    destination.mkdir(parents=True, exist_ok=False)
    journal = destination / 'archive_manifest.json'
    def save():
        temporary = journal.with_suffix('.tmp')
        temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
        temporary.replace(journal)
    save()
    for entry in plan['entries']:
        source, target = Path(entry['source']), Path(entry['destination'])
        if source.stat().st_dev != destination.stat().st_dev:
            raise ValueError('Cross-device operation forbidden')
        source.rename(target)
        if target.stat().st_ino != entry['inode']:
            raise RuntimeError('Unexpected inode change')
        entry['status'] = 'archived'
        save()
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    print(json.dumps(archive(execute=parser.parse_args().execute), ensure_ascii=False, indent=2))
