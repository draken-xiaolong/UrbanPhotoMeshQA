from collections import Counter
import numpy as np
import torch
from urbanphotomeshqa.v3_protocol import planned_slots, effective_rating, admission


def test_slot_balance_and_rotation():
    for index in range(30):
        rows = planned_slots(index)
        assert len(rows) == len({r['variant_id'] for r in rows}) == 33
        assert Counter(r['target_scale'] for r in rows) == {1:8,2:8,3:8,4:8,5:1}
    for combo in range(25,33):
        assert {planned_slots(i)[combo]['target_scale'] for i in range(4)} == {1,2,3,4}


def test_rating_provenance_and_no_target_fallback():
    machine = {'content_digest':'abc','scale':3,'uncertain':False,
               'evidence':['view1'], 'protocol_version':'test'}
    human = dict(machine, scale=2)
    assert effective_rating({'human':human,'machine':machine},'abc')['scale'] == 2
    assert effective_rating({'human':human,'machine':machine},'changed') is None
    assert effective_rating({'target_scale':5},'abc') is None
    assert not admission({'target_scale':3})['accepted']


def test_empty_patch_texture_loss_is_finite():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from train_local_patch_quality import loss_value, metrics
    prediction = {name:torch.full((1,16),.5,requires_grad=True)
                  for name in ('geometry','texture','overall')}
    data = {f'{name}_target':torch.ones((1,16)) for name in prediction}
    data.update(patch_mask=torch.ones((1,16),dtype=torch.bool),
                texture_supervision=torch.zeros((1,16),dtype=torch.bool))
    loss = loss_value(prediction,data,torch.tensor([0]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction['texture'].grad).all()
    assert metrics(np.zeros(2),np.zeros(2),np.zeros(2,dtype=bool))['mae'] is None
