"""Shared single-mesh local features for cache extraction and live inference."""

from __future__ import annotations
import numpy as np

from .patches import topological_patch_layout
from .texture import STANDARD_DIRECTIONS, render_textured_view_with_masks
from .texture_features import SpatialImageEncoder, patch_texture_atlases, texture_quality_statistics


def _padded(values, shape, dtype=None):
    output=np.zeros(shape,dtype=dtype or values.dtype)
    slices=tuple(slice(0,min(a,b)) for a,b in zip(values.shape,shape)); output[slices]=values[slices]
    return output


def extract_local_features(mesh, encoder: SpatialImageEncoder, render_size: int = 224, layout=None):
    """Return the exact arrays consumed by the v2 local no-reference head."""
    layout=topological_patch_layout(mesh,16) if layout is None else layout
    rendered=[render_textured_view_with_masks(mesh,direction,render_size,return_face_ids=True)
              for direction in STANDARD_DIRECTIONS]
    views=np.stack([v[0] for v in rendered]); foreground=np.stack([v[1] for v in rendered])
    textured=np.stack([v[2] for v in rendered]); face_ids=np.stack([v[3] for v in rendered])
    face_patch=layout["face_patch"]; safe=np.maximum(face_ids,0)
    masks=np.stack([textured & (face_ids>=0) & (face_patch[safe]==patch) for patch in range(16)])
    encoded=encoder.patch_tokens(views,masks)
    stats=np.stack([texture_quality_statistics(views,foreground & patch_mask,textured & patch_mask)
                    for patch_mask in masks]).astype(np.float32)
    atlas_images,atlas_masks=patch_texture_atlases(mesh,face_patch,render_size)
    atlas_encoded=encoder.masked_global_tokens(atlas_images,atlas_masks)
    atlas_stats=texture_quality_statistics(atlas_images,atlas_masks,atlas_masks)
    active=len(layout["descriptors"]); patch_mask=np.zeros(16,bool); patch_mask[:active]=True
    return {"patch_descriptors":_padded(layout["descriptors"],(16,58),np.float32),
            "patch_mask":patch_mask,"patch_area":_padded(layout["patch_area"],(16,),np.float32),
            "patch_center":_padded(layout["patch_center"],(16,3),np.float64),
            "patch_view_tokens":encoded["patch_view_tokens"],"patch_view_mask":encoded["patch_view_mask"],
            "patch_view_stats":stats,"patch_atlas_tokens":_padded(atlas_encoded["tokens"],(16,576),np.float16),
            "patch_atlas_mask":_padded(atlas_encoded["mask"],(16,),bool),
            "patch_atlas_stats":_padded(atlas_stats,(16,12),np.float32)}
