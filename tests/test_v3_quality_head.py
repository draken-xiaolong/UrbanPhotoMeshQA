import copy

import pytest
import torch

from urbanphotomeshqa.v3_quality_head import V3QualityHead, supervised_losses


def features():
    torch.manual_seed(2026)
    return torch.randn(2, 8), torch.randn(2, 16, 5), torch.ones((2, 16), dtype=torch.bool)


def empty_targets(output):
    labels = {}
    for level in ('building', 'patch'):
        for kind in ('quality', 'attribute'):
            name = f'{level}_{kind}'
            tensor = output[name if kind == 'quality' else name + '_logits']
            labels[name + '_target'] = torch.full_like(tensor, float('nan'))
            labels[name + '_valid'] = torch.zeros_like(tensor, dtype=torch.bool)
    labels['scale'] = torch.full((2,), float('nan'))
    labels['scale_valid'] = torch.zeros(2, dtype=torch.bool)
    return labels


def test_outputs_and_ordered_ordinal_logits():
    out = V3QualityHead(8, 5)(*features())
    assert out['building_attribute_logits'].shape == (2, 6)
    assert out['patch_attribute_logits'].shape == (2, 16, 6)
    assert out['patch_quality'].shape == (2, 16, 3)
    assert (out['ordinal_logits'][:, 1:] < out['ordinal_logits'][:, :-1]).all()


def test_invalid_nan_patches_do_not_contaminate_forward_or_loss():
    building, patches, valid = features()
    valid[0] = False
    patches[0] = float('nan')
    model = V3QualityHead(8, 5)
    out = model(building, patches, valid)
    assert all(torch.isfinite(x).all() for x in out.values())
    labels = empty_targets(out)
    # Even an erroneously valid supervision flag on a nonexistent slot is excluded.
    labels['patch_quality_valid'][0] = True
    losses = supervised_losses(out, labels)
    loss = sum(losses.values())
    assert loss.item() == 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_global_labels_do_not_supervise_unknown_local_quality():
    model = V3QualityHead(8, 5)
    out = model(*features())
    labels = empty_targets(out)
    labels['building_quality_target'][:, 2] = .5
    labels['building_quality_valid'][:, 2] = True
    labels['scale'][:] = 3
    labels['scale_valid'][:] = True
    labels['patch_attribute_valid'][0, 2, [0, 4]] = True
    labels['patch_attribute_target'][0, 2, [0, 4]] = 1
    losses = supervised_losses(out, labels)
    sum(losses.values()).backward()
    assert losses['patch_quality'].item() == 0
    assert torch.count_nonzero(model.patch_quality.weight.grad) == 0
    assert torch.count_nonzero(model.patch_attributes.weight.grad) > 0


def test_patch_reordering_is_equivariant_and_pooling_invariant():
    model = V3QualityHead(8, 5).eval()
    building, patches, valid = features()
    order = torch.randperm(16)
    first, second = model(building, patches, valid), model(building, patches[:, order], valid[:, order])
    torch.testing.assert_close(first['building_quality'], second['building_quality'])
    torch.testing.assert_close(first['patch_attribute_logits'][:, order], second['patch_attribute_logits'])


def test_checkpoint_roundtrip_and_legacy_schema_rejection():
    model = V3QualityHead(8, 5).eval()
    checkpoint = model.checkpoint('single_input_feature_v1_digest')
    restored = V3QualityHead.from_checkpoint(checkpoint, 'single_input_feature_v1_digest').eval()
    torch.testing.assert_close(model(*features())['building_quality'], restored(*features())['building_quality'])
    for field, value in [('attribute_order', ['Mixed'] * 6), ('patch_count', 32),
                         ('attribute_schema', 'old'), ('model_schema', 'old')]:
        bad = copy.copy(checkpoint)
        bad[field] = value
        with pytest.raises(ValueError):
            V3QualityHead.from_checkpoint(bad, 'single_input_feature_v1_digest')
    with pytest.raises(ValueError):
        V3QualityHead.from_checkpoint(checkpoint, 'different_extractor')


def test_nonfinite_valid_feature_and_out_of_range_scale_rejected():
    building, patches, valid = features()
    model = V3QualityHead(8, 5)
    patches[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        model(building, patches, valid)
    out = model(*features())
    labels = empty_targets(out)
    labels['scale'][0], labels['scale_valid'][0] = 3.5, True
    with pytest.raises(ValueError):
        supervised_losses(out, labels)
