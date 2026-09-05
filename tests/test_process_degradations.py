import numpy as np
from PIL import Image
from pathlib import Path
from urbanphotomeshqa.gltf import GltfReader
from urbanphotomeshqa.gltf import MeshAsset
from urbanphotomeshqa.process_degradations import SurfaceRegion, deform, surface_mask, local_texture, draco_positions


def mesh():
    v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,0,0],[1,1,0],[0,1,0]], float)
    return MeshAsset(v, np.array([[0,1,2],[3,4,5]]), np.tile([0.,0.,1.], (6,1)),
                     np.zeros(2,int), {}, v[:,:2]*.4+.1)


def test_nested_surface_area_and_shared_vertex_displacement():
    m = mesh(); region = SurfaceRegion(m, 2026)
    assert np.all(~region.mask(.2) | region.mask(.9))
    changed = deform(m, np.ones(2,bool), .1, 2026)
    assert np.array_equal(changed.vertices[0], changed.vertices[3])
    assert np.array_equal(changed.vertices[2], changed.vertices[4])


def test_multi_region_budget_is_shared_and_nested():
    from urbanphotomeshqa.process_degradations import MultiSurfaceRegion
    region = MultiSurfaceRegion(mesh(),2026,count=2)
    for fraction in (0,.01,.2,.9,1):
        weights = region.weights(fraction)
        assert np.isclose(np.dot(weights,region.area),fraction*region.area.sum())
    assert np.all(region.weights(.2) <= region.weights(.9))
    assert not SurfaceRegion(mesh(),2026).mask(0).any()


def test_texture_support_and_no_wrap():
    m = mesh(); mask = surface_mask(m, np.ones(2,bool), 0, (64,64))
    arr = np.random.default_rng(3).integers(0,255,(64,64,4), dtype=np.uint8)
    arr[...,3] = 255
    for kind in ('missing','seam','misalignment'):
        out = np.asarray(local_texture(Image.fromarray(arr),mask,kind,3 if kind=='misalignment' else .3))
        assert np.array_equal(out[~mask],arr[~mask])
    assert not mask[50:, :].any()


def test_ghost_does_not_disappear_when_shift_exceeds_destination_width():
    from urbanphotomeshqa.process_degradations import island_projection_ghost
    array = np.zeros((32,64,4),np.uint8)
    array[...,0] = np.arange(64,dtype=np.uint8)[None,:]*3
    array[...,3] = 193
    domain = np.ones((32,64),bool)
    destination = np.zeros_like(domain); destination[8:24,30:32] = True
    counts = []
    for shift in (.08,.18,.32,.48):
        out = np.asarray(island_projection_ghost(Image.fromarray(array),destination,domain,shift))
        counts.append(np.any(out[...,:3] != array[...,:3],axis=2).sum())
        assert np.array_equal(out[~destination],array[~destination])
        assert np.array_equal(out[...,3],array[...,3])
    assert counts == [32]*4


def test_ghost_never_samples_disconnected_island_or_transparent_background():
    from urbanphotomeshqa.process_degradations import island_projection_ghost
    array = np.zeros((24,32,4),np.uint8)
    array[2:20,2:10] = [255,0,0,255]
    array[2:20,20:28] = [0,255,0,255]
    domain = array[...,3] > 0
    out = np.asarray(island_projection_ghost(Image.fromarray(array),domain,domain,1))
    assert np.array_equal(out,array)


def test_repeat_uv_mask_is_translation_invariant():
    from dataclasses import replace
    m=mesh();selected=np.ones(2,bool)
    expected=surface_mask(m,selected,0,(64,64))
    tiled=replace(m,texcoords=m.texcoords+[-1,2])
    assert np.array_equal(surface_mask(tiled,selected,0,(64,64)),expected)
    crossing=replace(m,texcoords=m.texcoords+[-.3,0])
    mask=surface_mask(crossing,selected,0,(64,64))
    assert mask[:,:14].any() and mask[:,-14:].any()
    assert not mask[:,25:45].any()


def test_draco_order_keeps_uv_and_material_indices():
    m = mesh()
    out, payload = draco_positions(m, 10)
    assert payload
    assert np.array_equal(out.faces,m.faces)
    assert np.array_equal(out.texcoords,m.texcoords)


def test_textured_qem_retains_material_and_finite_uv():
    from urbanphotomeshqa.process_degradations import textured_qem
    fixture = Path(__file__).parent/'fixtures'/'B360011502301063A0'/'B360011502301063A0.gltf'
    original = GltfReader(fixture).load_mesh(include_texture=True)
    result = textured_qem(original, .6)
    assert 0 < len(result.faces) < len(original.faces)
    assert set(result.face_materials) == set(original.face_materials)
    assert np.isfinite(result.texcoords).all()
    assert np.isfinite(result.normals).all()
    assert np.all(result.vertices >= original.vertices.min(axis=0)-1e-6)
    assert np.all(result.vertices <= original.vertices.max(axis=0)+1e-6)


def test_qem_accepts_constant_color_faces_without_uv():
    from urbanphotomeshqa.process_degradations import textured_qem
    m=mesh();m.texcoords[:]=np.nan
    m.metadata['material_texture_paths']=[None]
    result=textured_qem(m,.6)
    assert np.isfinite(result.vertices).all()
    assert np.isfinite(result.texcoords).all()


def test_bounded_smoothing_caps_displacement_and_keeps_uv_seams_joined():
    m=mesh()
    out=deform(m,np.ones(2,bool),.25,2026,smooth_steps=30,max_displacement=.02)
    assert np.max(np.linalg.norm(out.vertices-m.vertices,axis=1)) <= .020000001
    assert np.array_equal(out.vertices[0],out.vertices[3])
    assert np.array_equal(out.texcoords,m.texcoords)


def test_explicit_surface_anchor_and_nested_levels():
    m=mesh()
    a=SurfaceRegion(m,2026,first_face=0)
    b=SurfaceRegion(m,2026,first_face=1)
    assert a.order[0]==0 and b.order[0]==1
    assert np.all(~a.mask(.1) | a.mask(.9))
