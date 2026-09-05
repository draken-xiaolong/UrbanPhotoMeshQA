import hashlib
import json

import pytest

from summarize_v3_blind_review import summarize


def setup_review(tmp_path):
    folder = tmp_path / 'public' / 'anonymous'
    folder.mkdir(parents=True)
    images = {}
    for name in [*(f'view{i}.png' for i in range(7)), 'views.jpg']:
        payload = name.encode()
        (folder / name).write_bytes(payload)
        images[name] = hashlib.sha256(payload).hexdigest()
    (folder / 'receipt.json').write_text(json.dumps({'content_digest': 'asset', 'images': images}))
    (tmp_path / 'review_queue.json').write_text(json.dumps([{'review_id': 'anonymous'}]))
    (tmp_path / 'private_mapping.json').write_text(json.dumps([{
        'review_id': 'anonymous', 'content_digest': 'asset', 'variant_id': 'T2_level4', 'target_scale': 1}]))
    scores = tmp_path / 'scores.json'
    scores.write_text(json.dumps({'status': 'independent_opinions_locked_before_mapping',
                                 'ratings': [{'review_id': 'anonymous', 'scale': 5, 'reason': 'No visible loss'}]}))
    return scores


def test_review_never_relabels_or_admits(tmp_path):
    scores = setup_review(tmp_path)
    original = scores.read_bytes()
    result = summarize(tmp_path, scores)
    assert result['target_matches'] == 0
    assert result['records'][0]['machine_scale'] == 5
    assert result['records'][0]['formal_admitted'] is False
    assert scores.read_bytes() == original


def test_review_rejects_modified_evidence(tmp_path):
    scores = setup_review(tmp_path)
    (tmp_path / 'public' / 'anonymous' / 'views.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError, match='evidence changed'):
        summarize(tmp_path, scores)


def test_review_rejects_duplicate_grades(tmp_path):
    scores = setup_review(tmp_path)
    payload = json.loads(scores.read_text())
    payload['ratings'] *= 2
    scores.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='exactly once'):
        summarize(tmp_path, scores)
