import copy

import numpy as np
import pytest
import torch

from urbanphotomeshqa.v3_attribute_labels import (
    empty_labels, compile_labels, masked_attribute_loss,
)


def review(value, source='machine_visual_review'):
    return dict(value=value, source=source, evidence=['view0.png#sha256'],
                protocol_version='visual_v2', uncertain=False)


def test_unknown_is_not_normal_and_multilabel_is_independent():
    doc = empty_labels('asset', 'layout')
    doc['building'][0] = review(1)
    doc['building'][3] = review(1)
    doc['patches'][2][0] = review(0, 'human')
    data = compile_labels(doc, 'asset', 'layout')
    assert data['building_valid'].sum() == 2
    assert data['building_target'].sum() == 2
    assert data['patches_valid'].sum() == 1
    assert data['patches_target'].sum() == 0


@pytest.mark.parametrize('mutation', [
    {'schema': 'old_softmax'}, {'attribute_order': ['Mixed'] * 6},
    {'content_digest': 'old'}, {'patch_layout_digest': 'old'},
    {'patches': [[None] * 6] * 32}, {'building': [0] * 6},
])
def test_stale_or_legacy_labels_rejected(mutation):
    doc = empty_labels('asset', 'layout')
    doc.update(mutation)
    with pytest.raises(ValueError):
        compile_labels(doc, 'asset', 'layout')


@pytest.mark.parametrize('entry', [review(True), review(float('nan')),
    review(1, 'intervention'), dict(review(1), uncertain=True),
    dict(review(0), evidence=[]), review(.8)])
def test_invalid_reviews_cannot_supply_supervision(entry):
    doc = empty_labels('asset', 'layout')
    doc['building'][0] = entry
    with pytest.raises(ValueError):
        compile_labels(doc, 'asset', 'layout')


def test_loss_ignores_unknown_nan_and_supports_all_unknown():
    logits = torch.zeros((1, 16, 6), requires_grad=True)
    targets = torch.full_like(logits, float('nan'))
    valid = torch.zeros_like(logits, dtype=torch.bool)
    loss = masked_attribute_loss(logits, targets, valid)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0
    valid[0, 3, 0] = valid[0, 3, 4] = True
    targets[valid] = 1
    logits.grad.zero_()
    loss = masked_attribute_loss(logits, targets, valid)
    assert loss.item() == pytest.approx(np.log(2))
    loss.backward()
    assert (logits.grad[valid] < 0).all()
    assert torch.count_nonzero(logits.grad[~valid]) == 0


def test_compile_does_not_mutate_review_document():
    doc = empty_labels('asset', 'layout')
    before = copy.deepcopy(doc)
    compile_labels(doc, 'asset', 'layout')
    assert doc == before
