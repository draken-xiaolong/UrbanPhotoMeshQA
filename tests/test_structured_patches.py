from types import SimpleNamespace

import numpy as np

from urbanphotomeshqa.structured_patches import structured_patch_layout


def mesh_from_triangles(triangles):
    vertices = np.asarray(triangles, np.float64).reshape(-1, 3)
    return SimpleNamespace(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3))


def test_distant_fragments_cannot_be_joined_to_fill_budget():
    mesh = mesh_from_triangles([[[i * 10, 0, 0], [i * 10 + 1, 0, 0], [i * 10, 1, 0]] for i in range(3)])
    out = structured_patch_layout(mesh, 2)
    assert len(out['virtual_bridge_faces']) == 0
    assert len(out['unassigned_face_indices']) == 1
    assert not out['full_valid_coverage']
    assert len(np.unique(out['real_component_id'])) == 3
    assert out['patch_mask'].tolist() == [True, True]


def test_nearby_links_obey_orientation_and_span():
    triangles = [[[0, 0, 0], [.01, 0, 0], [0, .01, 0]],
                 [[.012, 0, 0], [.022, 0, 0], [.012, .01, 0]],
                 [[1, 0, 0], [1, .01, 0], [1.01, 0, 0]]]
    mesh = mesh_from_triangles(triangles)
    out = structured_patch_layout(mesh, 2)
    assert len(out['virtual_bridge_faces']) == 1
    assert out['full_valid_coverage']
    assert (out['virtual_bridge_metrics'][:, 0] <= .025).all()
    assert (out['virtual_bridge_metrics'][:, 1] <= .35).all()
    assert (out['virtual_bridge_metrics'][:, 2] >= .5).all()
    reversed_mesh = mesh_from_triangles([triangles[0], triangles[1][::-1], triangles[2]])
    assert not len(structured_patch_layout(reversed_mesh, 2)['virtual_bridge_faces'])
    assert not len(structured_patch_layout(mesh, 2, bridge_span=.001)['virtual_bridge_faces'])


def test_fixed_slots_degenerate_face_mask_and_complete_valid_membership():
    mesh = mesh_from_triangles([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[2, 0, 0]] * 3])
    out = structured_patch_layout(mesh)
    assert out['descriptors'].shape == (16, 58)
    assert len(out['patch_offsets']) == 17
    assert out['patch_mask'].sum() == 1
    assert out['face_valid'].tolist() == [True, False]
    assert out['unassigned_face_indices'].tolist() == [1]
    assert out['full_valid_coverage']
    assert not out['formal_admitted']


def test_face_vertex_reordering_preserves_partition_and_membership():
    triangles = []
    for x in range(6):
        for y in range(4):
            triangles.extend([[[x,y,0], [x+1,y,0], [x,y+1,0]],
                              [[x+1,y,0], [x+1,y+1,0], [x,y+1,0]]])
    mesh = mesh_from_triangles(triangles)
    original = structured_patch_layout(mesh, 16)
    rng = np.random.default_rng(2026)
    fp = rng.permutation(len(mesh.faces)); vp = rng.permutation(len(mesh.vertices))
    inv = np.argsort(vp)
    reordered = SimpleNamespace(vertices=mesh.vertices[vp], faces=inv[mesh.faces[fp]])
    result = structured_patch_layout(reordered, 16)
    assert np.array_equal(result['face_patch'], original['face_patch'][fp])
    assert np.array_equal(np.sort(result['patch_face_indices']), np.arange(len(fp)))
    assert (result['patch_real_component_count'] == 1).all()
    assert result['patch_mask'].all()
