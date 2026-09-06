from types import SimpleNamespace
import numpy as np
from urbanphotomeshqa.uv_support import pixel_support_on_faces, patch_support


def mesh():
    return SimpleNamespace(vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[2,0,0],[3,0,0],[2,1,0]],float),
        faces=np.array([[0,1,2],[3,4,5]]), face_materials=np.array([0,0]),
        texcoords=np.array([[.1,.1],[.4,.1],[.1,.4]]*2), metadata={})


def test_shared_uv_propagates_to_both_disconnected_faces():
    m=mesh(); mask=np.zeros((100,100),bool);mask[:50,:50]=True
    np.testing.assert_array_equal(pixel_support_on_faces(m,{0:mask}),[1,1])
    # glTF top row, not the legacy renderer's flipped V convention.
    np.testing.assert_array_equal(pixel_support_on_faces(m,{0:mask[::-1]}),[0,0])


def test_repeat_and_mirror_are_evaluated_per_sample():
    m=mesh();m.texcoords+=1
    mask=np.zeros((100,100),bool);mask[:50,:50]=True
    np.testing.assert_array_equal(pixel_support_on_faces(m,{0:mask}),[1,1])
    m.metadata={'material_profiles':[{'baseColorSampler':{'wrapS':33648,'wrapT':33648}}]}
    np.testing.assert_array_equal(pixel_support_on_faces(m,{0:mask}),[0,0])


def test_unknown_material_is_not_normal_and_patch_uses_known_area():
    m=mesh();m.face_materials[1]=1
    f=pixel_support_on_faces(m,{0:np.ones((10,10),bool)})
    assert f[0]==1 and np.isnan(f[1])
    result=patch_support(m,np.array([0,0]),f)
    assert result['support_fraction_of_known_area'][0]==1
    assert result['known_area_fraction'][0]==.5
    assert np.isnan(result['support_fraction_of_known_area'][1:]).all()
