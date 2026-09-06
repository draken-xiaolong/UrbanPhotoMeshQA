"""Approximate surface support of actual texture pixels, including UV reuse.

This is intervention evidence, NOT a visible-defect or local-quality label.
Inputs use native glTF UVs (V=0 at the top image row), not legacy renderer UVs.
"""
import numpy as np
from .texture import _wrap_texture_coordinate


def pixel_support_on_faces(mesh, masks, samples=1024):
    """Sample every face using each material's mask, including unselected UV copies.

    Deterministic area-uniform points within each 3D triangle. Degenerate UVs
    still address texture pixels; degenerate 3D faces remain unknown. This is
    bounded numerical quadrature, not an exact area intersection algorithm.
    """
    if samples not in (256, 1024, 4096):
        raise ValueError('Use a bounded quadrature resolution')
    if mesh.texcoords is None:
        raise ValueError('Native glTF UVs required')
    # Stratified square grid mapped area-uniformly to barycentric coordinates.
    n = int(np.sqrt(samples))
    x, y = np.meshgrid((np.arange(n)+.5)/n, (np.arange(n)+.5)/n)
    s = np.sqrt(x.ravel())
    bary = np.stack((1-s, s*(1-y.ravel()), s*y.ravel()), axis=1)
    result = np.full(len(mesh.faces), np.nan)
    profiles = mesh.metadata.get('material_profiles', [])
    triangles = mesh.vertices[mesh.faces]
    area2 = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1)
    for material, raw_mask in masks.items():
        mask = np.asarray(raw_mask, bool)
        if mask.ndim != 2 or min(mask.shape) < 1:
            raise ValueError('Pixel masks must be non-empty 2D arrays')
        profile = profiles[material] if material < len(profiles) else {}
        sampler = profile.get('baseColorSampler') or {}
        ws, wt = sampler.get('wrapS', 10497), sampler.get('wrapT', 10497)
        if ws not in (10497, 33071, 33648) or wt not in (10497, 33071, 33648):
            raise ValueError('Unsupported wrap mode')
        for face in np.flatnonzero((mesh.face_materials == material) & (area2 > 1e-15)):
            uv = mesh.texcoords[mesh.faces[face]]
            if not np.isfinite(uv).all():
                continue
            points = bary@uv
            u = _wrap_texture_coordinate(points[:,0], ws)
            v = _wrap_texture_coordinate(points[:,1], wt)
            px = np.rint(u*(mask.shape[1]-1)).astype(int)
            py = np.rint(v*(mask.shape[0]-1)).astype(int)
            result[face] = mask[py, px].mean()
    return result


def patch_support(mesh, face_patch, fractions, count=16):
    """Area-weighted support and known-area fraction; unknown does not become zero."""
    labels = np.asarray(face_patch)
    values = np.asarray(fractions)
    if labels.shape != (len(mesh.faces),) or values.shape != labels.shape:
        raise ValueError('Face support/layout shape mismatch')
    if count != 16 or np.any(labels < 0) or np.any(labels >= count):
        raise ValueError('A current-asset fixed16 layout is required')
    tri = mesh.vertices[mesh.faces]
    area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)/2
    known = np.isfinite(values)
    total = np.bincount(labels, weights=area, minlength=count)
    covered = np.bincount(labels, weights=area*known, minlength=count)
    numerator = np.bincount(labels, weights=area*np.where(known,values,0), minlength=count)
    return {'support_fraction_of_known_area': np.divide(numerator, covered, out=np.full(count,np.nan), where=covered>0),
            'known_area_fraction': np.divide(covered,total,out=np.zeros(count),where=total>0)}
