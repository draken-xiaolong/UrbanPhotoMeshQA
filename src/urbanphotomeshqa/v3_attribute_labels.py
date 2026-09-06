"""Versioned visible-attribute targets, separate from recipe intervention masks.

Unknown entries stay null on disk and have zero loss weight in memory. Neither
Clean status nor an applied recipe supplies negative/positive visible labels.
"""
import math

import numpy as np

from .v3_protocol import ATTRIBUTES, PATCH_COUNT

SCHEMA = 'v3_visible_attributes_v1'


def empty_labels(content_digest, patch_layout_digest):
    if not content_digest or not patch_layout_digest:
        raise ValueError('Content and current patch layout digests are required')
    return {
        'schema': SCHEMA, 'attribute_order': list(ATTRIBUTES),
        'content_digest': content_digest, 'patch_layout_digest': patch_layout_digest,
        'building': [None] * len(ATTRIBUTES),
        'patches': [[None] * len(ATTRIBUTES) for _ in range(PATCH_COUNT)],
    }


def _entry(entry):
    if entry is None:
        return 0., False
    if not isinstance(entry, dict):
        raise ValueError('Labels must be reviewed entries or null, not recipe IDs')
    value = entry.get('value')
    if type(value) not in (int, float) or not math.isfinite(value) or value not in (0, 1):
        raise ValueError('Visible presence must be explicitly reviewed binary 0/1')
    if entry.get('source') not in ('human', 'machine_visual_review'):
        raise ValueError('Intervention masks are not visible-attribute supervision')
    if not entry.get('evidence') or not entry.get('protocol_version'):
        raise ValueError('Reviewed labels require evidence and protocol provenance')
    if entry.get('uncertain') is not False:
        raise ValueError('Unresolved labels must remain null')
    return float(value), True


def compile_labels(document, content_digest, patch_layout_digest):
    """Fail closed on stale content/layout, old classes, or ambiguous negatives."""
    if document.get('schema') != SCHEMA or document.get('attribute_order') != list(ATTRIBUTES):
        raise ValueError('Incompatible visible-attribute schema/order')
    if not content_digest or document.get('content_digest') != content_digest:
        raise ValueError('Stale content labels')
    if not patch_layout_digest or document.get('patch_layout_digest') != patch_layout_digest:
        raise ValueError('Stale current-mesh patch layout')
    result = {}
    for name, shape in (('building', (6,)), ('patches', (PATCH_COUNT, 6))):
        array = np.asarray(document.get(name), dtype=object)
        if array.shape != shape:
            raise ValueError(f'{name}: expected shape {shape}, got {array.shape}')
        target, valid = np.zeros(shape, np.float32), np.zeros(shape, bool)
        for index in np.ndindex(shape):
            target[index], valid[index] = _entry(array[index])
        result[name + '_target'] = target
        result[name + '_valid'] = valid
    return result


def masked_attribute_loss(logits, targets, valid):
    """Independent sigmoid BCE; exclude unknowns before evaluating their values."""
    import torch
    import torch.nn.functional as F

    if logits.shape != targets.shape or logits.shape != valid.shape or logits.shape[-1] != 6:
        raise ValueError('Expected aligned six-attribute tensors')
    if valid.dtype != torch.bool:
        raise ValueError('Validity mask must be boolean')
    selected_logits, selected_targets = logits[valid], targets[valid]
    if not torch.isfinite(selected_logits).all() or not torch.isfinite(selected_targets).all():
        raise ValueError('Non-finite supervised values')
    if not ((selected_targets == 0) | (selected_targets == 1)).all():
        raise ValueError('Supervised targets must be binary')
    if not selected_logits.numel():
        return selected_logits.sum()  # Differentiable zero, including masked NaNs.
    return F.binary_cross_entropy_with_logits(selected_logits, selected_targets)
