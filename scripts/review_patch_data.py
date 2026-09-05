"""Source intervention display mapping. QEM/current meshes use approximate nearest faces."""
from functools import lru_cache
from pathlib import Path
import sys
import json
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from urbanphotomeshqa.gltf import GltfReader


@lru_cache(maxsize=60)
def patch_data(root_string, variant):
    root = Path(root_string)
    records = {r['variant']:r for r in json.loads((root/'manifest.json').read_text())['records']}
    clean = GltfReader(records['clean']['gltf_path']).load_mesh(include_texture=True)
    mesh = GltfReader(records[variant]['gltf_path']).load_mesh(include_texture=True)
    with np.load(root/'source_patch_layout.npz') as layout:
        labels = layout['face_patch'].copy()
    with np.load(root/'assets'/variant/'face_support.npz') as support:
        geometry = support['geometry'].copy(); texture = support['texture'].copy(); area = support['source_face_areas'].copy()
    centers = clean.vertices[clean.faces].mean(axis=1)
    current = mesh.vertices[mesh.faces].mean(axis=1)
    distance, mapping = cKDTree(centers).query(current)
    same = (len(mesh.faces)==len(clean.faces) and np.array_equal(mesh.faces,clean.faces))
    # Export can reorder faces by material; only use direct IDs when coordinates match too.
    exact = same and np.allclose(current,centers,atol=1e-5,rtol=0)
    if exact: mapping=np.arange(len(labels)); distance=np.zeros(len(labels))
    total = np.bincount(labels,weights=area,minlength=16)
    stats = [{'patch':i+1,'geometry_fraction':float(np.sum(area[(labels==i)&geometry])/max(total[i],1e-12)),
              'texture_fraction':float(np.sum(area[(labels==i)&texture])/max(total[i],1e-12))} for i in range(16)]
    return {'variant':variant,'source':{'patch':labels.tolist(),'geometry':geometry.astype(int).tolist(),'texture':texture.astype(int).tolist()},
            'current':{'patch':labels[mapping].tolist(),'geometry':geometry[mapping].astype(int).tolist(),'texture':texture[mapping].astype(int).tolist()},
            'stats':stats,'mapping':'exact' if exact else 'nearest_source_face_approximation',
            'max_mapping_distance':float(distance.max()),
            'note':'生成干预范围；不是模型预测或感知质量真值。缺失面位置在Clean侧查看；变形/简化后的映射为近似。'}
