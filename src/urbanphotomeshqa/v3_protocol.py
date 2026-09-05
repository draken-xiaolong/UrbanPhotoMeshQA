"""V3 admission contracts. Target grades and interventions are never truth labels."""
import hashlib

VERSION = 'v3_6class33_seed2026'
PATCH_COUNT = 16
ATTRIBUTES = ('G1', 'G2', 'G3', 'T1', 'T2', 'T3')
COMBINATIONS = (('G1','T3'), ('G2','T2'), ('G3','T1'), ('T1','T2'),
                ('T2','T3'), ('G1','G2'), ('G3','T1','T2'), ('G1','G2','T3'))
SPLIT_BUILDINGS = {'train':72, 'val':24, 'test':24, 'blind':30}
MAX_ATTEMPTS = 6


def planned_slots(building_index):
    """Index is assigned inside each split, never derived from model performance."""
    records = [{'variant_id':'clean', 'target_scale':5, 'applied_classes':[]}]
    for name in ATTRIBUTES:
        for level in range(1,5):
            records.append({'variant_id':f'{name}_level{level}', 'level':level,
                            'target_scale':5-level, 'applied_classes':[name]})
    for i, classes in enumerate(COMBINATIONS):
        records.append({'variant_id':f'C{i+1}', 'target_scale':1+(i+building_index)%4,
                        'applied_classes':list(classes)})
    return records


def stable_seed(asset_id, recipe_id):
    # Deliberately exclude level so the four targets share region ordering.
    return int.from_bytes(hashlib.sha256(f'2026:{asset_id}:{recipe_id}'.encode()).digest()[:4], 'big')


def effective_rating(ratings, digest):
    """Ignore stale or unresolved labels; do not fall back to target_scale."""
    for source in ('human', 'machine'):
        rating = ratings.get(source)
        if not rating or rating.get('content_digest') != digest:
            continue
        if rating.get('uncertain', True) or not rating.get('evidence'):
            continue
        scale = rating.get('scale')
        if type(scale) is not int or scale not in range(1,6):
            continue
        if not rating.get('protocol_version'):
            continue
        return {'scale':scale, 'score':(scale-1)/4, 'source':source}
    return None


def admission(record, known_digests=()):
    """Fail closed; incomplete candidates cannot silently enter formal manifests."""
    reasons = []
    digest = record.get('content_digest')
    if not digest or digest in known_digests:
        reasons.append('missing_or_duplicate_content')
    if record.get('technical_valid') is not True:
        reasons.append('technical_not_verified')
    if record.get('clean_admission_scale') != 5:
        reasons.append('clean_not_scale5')
    if record.get('physical_plausibility_verified') is not True:
        reasons.append('plausibility_not_verified')
    if record.get('patch_count') != PATCH_COUNT:
        reasons.append('patch_contract_mismatch')
    rating = effective_rating(record.get('ratings', {}), digest)
    if rating is None:
        reasons.append('no_independent_rating')
    elif rating['scale'] != record.get('target_scale'):
        reasons.append('target_observation_mismatch')
    return {'accepted':not reasons, 'reasons':reasons, 'effective_rating':rating}
