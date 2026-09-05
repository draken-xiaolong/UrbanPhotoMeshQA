"""Persist explicit AI visual opinions after inspection; never infer grade from geometry."""
import json
from pathlib import Path
from urbanphotomeshqa.integrity import asset_digest, sha256_file

ROOT = Path('/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/HK3D-Individualised-V3')
SOURCE = Path('/Volumes/SANDISK-ELE/HK3D-Individualised')
EVIDENCE = ROOT/'_archive/pre_6class33_20260905T201731Z/_review/clean_reselection_20260906'
REVIEWS = {
    'B415722108801063A0': {
        'role':'development_calibration',
        'reason':'七视角轮廓、主要屋顶结构和窗格排列稳定；正常尺度无明显大面积模糊、拉伸或重影。两张896像素立面复核后仍评5。基座纯色面与立面色带保留记录，不能仅凭纯色推断缺纹理。',
        'detail_views':['view0.png','view3.png']},
    'B421142104901063A0': {
        'role':'reviewed_reserve_train_only',
        'reason':'七视角阳台、窗框和屋顶边缘连续；896像素正面窗格细节可辨。复杂石材纹样及栏杆密集不是自动降分依据，正常尺度无突出映射破坏。与A形态相似，暂不占第二个多样性校准名额。',
        'detail_views':['view0.png']},
}


def main():
    if not Path('/Volumes/SANDISK-ELE').is_mount():
        raise RuntimeError('请先插上移动硬盘 SANDISK-ELE')
    rows=json.loads(Path('artifacts/manifests/iteration2_source_audit_seed2026.json').read_text())['records']
    for row in rows:
        review=REVIEWS.get(row['asset_id'])
        if review is None:
            continue
        path=SOURCE/row['source_gltf']
        digest,dependencies=asset_digest(path)
        if digest != row['asset_digest']:
            raise ValueError('Source changed; archived visual evidence cannot be reused')
        evidence=[]
        for name in ['views.jpg',*review['detail_views']]:
            image=EVIDENCE/row['asset_id']/name
            evidence.append({'path':str(image.relative_to(ROOT)), 'sha256':sha256_file(image)})
        rating={'content_digest':digest,'scale':5,'uncertain':False,
                'protocol_version':'v3_visual_admission_20260906_draft',
                'evidence':evidence,'reason':review['reason'],
                'reviewer':'assistant_visual_review','human_mos':False,
                'limitation':'7离线视角＋指定近观；未做实时整圈旋转。并非测绘精度/纹理完美性认证。'}
        result={'asset_id':row['asset_id'],'source_gltf':row['source_gltf'],
                'role':review['role'],'technical_valid':row['status']=='qualified',
                'technical_evidence':'Existing audit reused after full package hash equality',
                'source_dependencies':dependencies,'ratings':{'machine':rating},
                'formal_split':None,'protocol_frozen':False}
        target=ROOT/'ratings/clean_admission_20260906'/f"{row['asset_id']}.json"
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            raise FileExistsError('Do not overwrite previous review')
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(row['asset_id'],review['role'],digest,flush=True)


if __name__=='__main__':
    main()
