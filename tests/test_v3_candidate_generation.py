import json
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_v3_33_candidates import build, recipe
from urbanphotomeshqa.v3_protocol import planned_slots
from urbanphotomeshqa.integrity import asset_digest


def test_recipe_stable_across_grade_and_combo_subtypes():
    for index in (0,1):
        records=planned_slots(index)
        for category in ('G1','G2','G3','T1','T2','T3'):
            variants=[s for s in records if s['variant_id'].startswith(category)]
            assert len({recipe(s,index,100)[0]['kind'] for s in variants}) == 1
        for slot in records:
            ops=recipe(slot,index,100)
            assert len(ops)==len(slot['applied_classes'])
            if slot['variant_id'] in ('C1','C8'):
                assert next(o for o in ops if o['class']=='T3')['kind']=='missing_texture'
            if slot['variant_id']=='C5':
                assert next(o for o in ops if o['class']=='T3')['kind']=='radiometric'


@pytest.mark.parametrize('variant',['T1_level1','C1','C2','C3','C4','C5','C6','C7','C8'])
def test_fixture_candidate_has_no_automatic_quality_truth(tmp_path,variant,monkeypatch):
    source=Path(__file__).parent/'fixtures/B360011502301063A0/B360011502301063A0.gltf'
    digest,_=asset_digest(source)
    admission=tmp_path/'test_admission.json'
    # Test-only synthetic admission, never placed in a production dataset.
    admission.write_text(json.dumps({'technical_valid':True,'ratings':{'machine':{
        'content_digest':digest,'scale':5,'uncertain':False,'protocol_version':'unit_test_only',
        'evidence':['synthetic test harness; not a visual review']}}}))
    rows=build(source,tmp_path/'candidate',admission,0,[variant])
    assert len(rows)==1 and rows[0]['ratings']=={}
    assert rows[0]['formal_admitted'] is False
    assert not np.any(rows[0]['patch_quality_valid_mask'])
    from urbanphotomeshqa.v3_attribute_labels import compile_labels
    labels=json.loads((tmp_path/'candidate/assets'/variant/'visible_attribute_labels.json').read_text())
    targets=compile_labels(labels,rows[0]['content_digest'],rows[0]['patch_layout_digest'])
    assert not targets['building_valid'].any()
    assert not targets['patches_valid'].any()
    from audit_v3_candidates import audit
    report=audit(tmp_path/'candidate')
    assert report['count']==1 and report['passed']
    assert report['formal_admitted'] is False
    labels_path=tmp_path/'candidate/assets'/variant/'visible_attribute_labels.json'
    stale=dict(labels,patch_layout_digest='stale')
    labels_path.write_text(json.dumps(stale))
    with pytest.raises(ValueError,match='patch layout'):
        audit(tmp_path/'candidate')
    labels_path.write_text(json.dumps(labels))
    support=tmp_path/'candidate/assets'/variant/'intervention_support.npz'
    with np.load(support) as data:
        assert data['source_face_attributes'].shape[1]==6
        assert not np.any(data['source_face_attributes'][~data['source_face_retained'],3:])
    if variant=='T1_level1':
        from render_v3_blind_evidence import main as render_evidence
        monkeypatch.setattr(sys,'argv',['render','--root',str(tmp_path/'candidate'),
                                      '--output',str(tmp_path/'evidence'),'--size','32'])
        render_evidence()
        queue=json.loads((tmp_path/'evidence/review_queue.json').read_text())
        assert len(queue)==1 and 'target_scale' not in queue[0]
        assert 'T1' not in json.dumps(queue)
        image=tmp_path/'evidence'/queue[0]['views']
        timestamp=image.stat().st_mtime_ns
        render_evidence()
        assert image.stat().st_mtime_ns==timestamp
    admission.write_text('{}')
    with pytest.raises(ValueError,match='scale5'):
        build(source,tmp_path/'refused',admission,0,['T1_level1'])
    assert not (tmp_path/'refused').exists()
