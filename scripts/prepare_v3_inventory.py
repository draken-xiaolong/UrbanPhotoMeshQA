"""Lightweight source inventory and pending admission queue, not a quality audit."""
import json
from pathlib import Path
from collections import Counter
from urbanphotomeshqa.v3_protocol import VERSION, planned_slots

SOURCE = Path('/Volumes/SANDISK-ELE/HK3D-Individualised')
ROOT = Path('/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/HK3D-Individualised-V3')


def main():
    if not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('请先插上移动硬盘 SANDISK-ELE')
    # All ten currently unpacked tiles were used by previous project iterations.
    seen = {'11-NE-13B','11-NE-14A','11-SW-10C','11-SW-14B','11-SW-15A',
            '11-SW-3B','11-SW-4B','11-SW-4D','11-SW-5A','11-SW-9D'}
    records = []
    for tile in sorted(SOURCE.iterdir()):
        building = tile / 'BUILDING'
        if not tile.is_dir() or not building.is_dir():
            continue
        for path in sorted(building.glob('*/*.gltf')):
            if path.name.startswith('._'):
                continue
            records.append({'asset_id':path.parent.name, 'tile':tile.name,
                            'source_gltf':str(path.relative_to(SOURCE)),
                            'historically_seen_tile':tile.name in seen,
                            'clean_admission':'pending_independent_visual_review',
                            'scale':None, 'split':None})
    directory = ROOT / 'manifests' / VERSION
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / 'source_inventory.json'
    if output.exists():
        raise FileExistsError('Inventory exists; reuse it or explicitly version changes')
    payload = {'schema_version':1,'dataset_version':VERSION,'source_root':str(SOURCE),
               'counts_by_tile':dict(Counter(r['tile'] for r in records)),
               'formal_admitted_count':0,'records':records,
               'blind_status':'New untouched tiles required; do not relabel previously seen tiles.'}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (directory / 'slot_template.json').write_text(json.dumps(planned_slots(0), indent=2))
    print(json.dumps({k:v for k,v in payload.items() if k!='records'},ensure_ascii=False))


if __name__ == '__main__':
    main()
