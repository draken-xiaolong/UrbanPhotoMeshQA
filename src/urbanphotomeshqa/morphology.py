from __future__ import annotations

import torch


def _normalised_eigenvalues(covariance: torch.Tensor) -> torch.Tensor:
    values = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def global_morphology_targets(points: torch.Tensor) -> torch.Tensor:
    """Rotation/translation/scale-invariant global shape statistics.

    Args:
        points: ``[B, N, 6]`` normalised XYZ and unit normals.

    Returns:
        ``[B, 13]`` containing positional spectrum, sorted bounding-box ratios,
        radial quantiles, and an unoriented normal-distribution spectrum.
    """
    xyz = points[..., :3]
    normals = points[..., 3:6]
    centered = xyz - xyz.mean(dim=1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered / max(xyz.shape[1] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    positional_spectrum = eigenvalues.clamp_min(0.0)
    positional_spectrum = positional_spectrum / positional_spectrum.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # Extents in the PCA frame are invariant to the input coordinate frame.
    aligned = centered @ eigenvectors
    extents = aligned.amax(dim=1) - aligned.amin(dim=1)
    extents = torch.sort(extents, dim=1).values
    extent_ratios = extents / extents[:, -1:].clamp_min(1e-8)

    radius = torch.linalg.vector_norm(centered, dim=-1)
    radial_quantiles = torch.quantile(
        radius, torch.tensor([0.25, 0.5, 0.75, 0.95], device=points.device), dim=1
    ).transpose(0, 1)
    radial_quantiles = radial_quantiles / radial_quantiles[:, -1:].clamp_min(1e-8)

    # n n^T is invariant to a global sign flip of surface normals.
    normal_second_moment = normals.transpose(1, 2) @ normals / max(normals.shape[1], 1)
    normal_spectrum = _normalised_eigenvalues(normal_second_moment)
    return torch.cat(
        [positional_spectrum, extent_ratios, radial_quantiles, normal_spectrum], dim=1
    )


def local_morphology_targets(points: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Invariant local neighbourhood morphology for every sampled surface point.

    The seven targets are the normalised covariance spectrum, mean/std radial
    neighbour distance, and mean/std absolute normal agreement.
    """
    xyz = points[..., :3]
    normals = points[..., 3:6]
    count = xyz.shape[1]
    effective_k = min(k + 1, count)
    distances = torch.cdist(xyz, xyz)
    indices = distances.topk(effective_k, largest=False).indices[..., 1:]
    if indices.shape[-1] == 0:
        indices = distances.topk(1, largest=False).indices
    batch = torch.arange(xyz.shape[0], device=points.device)[:, None, None]
    neighbours = xyz[batch, indices]
    delta = neighbours - xyz[:, :, None, :]
    covariance = delta.transpose(-1, -2) @ delta / max(indices.shape[-1], 1)
    spectrum = _normalised_eigenvalues(covariance)

    radial = torch.linalg.vector_norm(delta, dim=-1)
    radial_mean = radial.mean(dim=-1, keepdim=True)
    radial_std = radial.std(dim=-1, keepdim=True, unbiased=False)

    neighbour_normals = normals[batch, indices]
    agreement = torch.abs((neighbour_normals * normals[:, :, None, :]).sum(dim=-1))
    normal_mean = agreement.mean(dim=-1, keepdim=True)
    normal_std = agreement.std(dim=-1, keepdim=True, unbiased=False)
    return torch.cat([spectrum, radial_mean, radial_std, normal_mean, normal_std], dim=-1)
