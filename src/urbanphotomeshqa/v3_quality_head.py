"""V3 lightweight shared head; no clean, recipe, or label is a forward input.

This is a composable baseline, not a trained/released predictor. Features must be
produced by the same versioned single-asset extractor during training/inference.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .v3_attribute_labels import SCHEMA as ATTRIBUTE_SCHEMA, masked_attribute_loss
from .v3_protocol import ATTRIBUTES, PATCH_COUNT

MODEL_SCHEMA = 'v3_joint_quality_head_v1'
QUALITY_ORDER = ('geometry', 'texture', 'overall')


class V3QualityHead(nn.Module):
    def __init__(self, building_dim, patch_dim, hidden_dim=128):
        super().__init__()
        for value in (building_dim, patch_dim, hidden_dim):
            if type(value) is not int or value < 1:
                raise ValueError('Feature dimensions must be positive integers')
        self.config = dict(building_dim=building_dim, patch_dim=patch_dim, hidden_dim=hidden_dim)
        self.patch_encoder = nn.Sequential(nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim), nn.GELU())
        self.building_encoder = nn.Sequential(nn.LayerNorm(building_dim),
            nn.Linear(building_dim, hidden_dim), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU())
        self.building_quality = nn.Linear(hidden_dim, 3)
        self.patch_quality = nn.Linear(hidden_dim, 3)
        self.building_attributes = nn.Linear(hidden_dim, 6)
        self.patch_attributes = nn.Linear(hidden_dim, 6)
        self.ordinal_latent = nn.Linear(hidden_dim, 1)
        self.ordinal_start = nn.Parameter(torch.tensor(-1.5))
        self.ordinal_gaps = nn.Parameter(torch.zeros(3))

    def forward(self, building_features, patch_features, patch_valid):
        batch = building_features.shape[0]
        if building_features.shape != (batch, self.config['building_dim']):
            raise ValueError('Invalid building feature dimensions')
        if patch_features.shape != (batch, PATCH_COUNT, self.config['patch_dim']):
            raise ValueError('Expected current-mesh features for exactly 16 patches')
        if patch_valid.shape != (batch, PATCH_COUNT) or patch_valid.dtype != torch.bool:
            raise ValueError('Expected boolean current-patch validity mask')
        if not torch.isfinite(building_features).all() or not torch.isfinite(patch_features[patch_valid]).all():
            raise ValueError('Non-finite valid features')
        # Select before encoding: masked NaNs must not contaminate LayerNorm or pooling.
        patches = self.patch_encoder(patch_features.masked_fill(~patch_valid[..., None], 0))
        patches = patches.masked_fill(~patch_valid[..., None], 0)
        pooled = patches.sum(1) / patch_valid.sum(1, keepdim=True).clamp_min(1)
        building = self.fusion(torch.cat((self.building_encoder(building_features), pooled), -1))
        cuts = self.ordinal_start + torch.cat((self.ordinal_gaps.new_zeros(1),
                                               F.softplus(self.ordinal_gaps).cumsum(0)))
        return {
            'building_quality': torch.sigmoid(self.building_quality(building)),
            'patch_quality': torch.sigmoid(self.patch_quality(patches)),
            'building_attribute_logits': self.building_attributes(building),
            'patch_attribute_logits': self.patch_attributes(patches),
            'ordinal_logits': self.ordinal_latent(building) - cuts,
            'patch_valid': patch_valid,
        }

    def checkpoint(self, feature_signature):
        if not isinstance(feature_signature, str) or not feature_signature:
            raise ValueError('A versioned single-input feature signature is required')
        return dict(model_schema=MODEL_SCHEMA, attribute_schema=ATTRIBUTE_SCHEMA,
                    attribute_order=list(ATTRIBUTES), quality_order=list(QUALITY_ORDER),
                    patch_count=PATCH_COUNT, config=self.config.copy(),
                    feature_signature=feature_signature, state_dict=self.state_dict())

    @classmethod
    def from_checkpoint(cls, checkpoint, feature_signature):
        expected = dict(model_schema=MODEL_SCHEMA, attribute_schema=ATTRIBUTE_SCHEMA,
                        attribute_order=list(ATTRIBUTES), quality_order=list(QUALITY_ORDER),
                        patch_count=PATCH_COUNT, feature_signature=feature_signature)
        if not feature_signature or any(checkpoint.get(k) != v for k, v in expected.items()):
            raise ValueError('Incompatible V3 model/labels/features; legacy checkpoints cannot be reused silently')
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        return model


def _masked_quality(prediction, target, valid):
    if prediction.shape != target.shape or prediction.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError('Quality target and boolean mask shape mismatch')
    prediction, target = prediction[valid], target[valid]
    if not torch.isfinite(target).all() or not ((target >= 0) & (target <= 1)).all():
        raise ValueError('Valid quality targets must be finite values in [0,1]')
    return F.smooth_l1_loss(prediction, target) if prediction.numel() else prediction.sum()


def supervised_losses(output, labels):
    """Return separate objectives; the training protocol must set explicit weights.

    All targets/masks are required. Unknown quality is NOT inferred from global
    grades, modalities, intervention area, or another quality prediction.
    """
    result = {}
    patch_valid = output['patch_valid']
    for level in ('building', 'patch'):
        for kind in ('quality', 'attribute'):
            name = f'{level}_{kind}'
            valid = labels[name + '_valid']
            prediction = output[name if kind == 'quality' else name + '_logits']
            if valid.shape != prediction.shape or valid.dtype != torch.bool:
                raise ValueError(f'{name}: invalid supervision mask')
            if level == 'patch':
                valid = valid & patch_valid[..., None]
            loss = _masked_quality if kind == 'quality' else masked_attribute_loss
            result[name] = loss(prediction, labels[name + '_target'], valid)
    scale, valid = labels['scale'], labels['scale_valid']
    if scale.shape != output['ordinal_logits'].shape[:1] or valid.shape != scale.shape or valid.dtype != torch.bool:
        raise ValueError('Invalid scale/mask shape')
    selected = scale[valid]
    if not torch.isfinite(selected).all() or not ((selected >= 1) & (selected <= 5) & (selected == selected.round())).all():
        raise ValueError('Valid ordinal scales must be integers 1..5')
    logits = output['ordinal_logits'][valid]
    truth = (selected[:, None] > torch.arange(1, 5, device=scale.device)).to(logits.dtype)
    result['ordinal'] = F.binary_cross_entropy_with_logits(logits, truth) if selected.numel() else logits.sum()
    return result
