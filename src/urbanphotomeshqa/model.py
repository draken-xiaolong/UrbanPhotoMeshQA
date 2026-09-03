from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: [B, C, N]. Pairwise squared distance without materialising broadcast tensors.
    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)
    square = torch.sum(x**2, dim=1, keepdim=True)
    pairwise = -square - inner - square.transpose(2, 1)
    return pairwise.topk(k=min(k, x.shape[-1]), dim=-1).indices


def graph_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    batch, channels, points = x.shape
    indices = knn(x, k)
    effective_k = indices.shape[-1]
    device = x.device
    offsets = torch.arange(batch, device=device).view(-1, 1, 1) * points
    flat_indices = (indices + offsets).reshape(-1)
    transposed = x.transpose(2, 1).contiguous()
    neighbors = transposed.reshape(batch * points, channels)[flat_indices]
    neighbors = neighbors.view(batch, points, effective_k, channels)
    centers = transposed.view(batch, points, 1, channels).expand(-1, -1, effective_k, -1)
    return torch.cat([neighbors - centers, centers], dim=3).permute(0, 3, 1, 2).contiguous()


class EdgeConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, k: int):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(input_channels * 2, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(graph_feature(x, self.k)).max(dim=-1).values


class BuildingBaseEncoder(nn.Module):
    """Shared geometry encoder with global identity, local and quality-sensitive outputs."""

    def __init__(
        self,
        input_dim: int = 6,
        embedding_dim: int = 256,
        local_dim: int = 128,
        k: int = 20,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.edge1 = EdgeConv(input_dim, 64, k)
        self.edge2 = EdgeConv(64, 64, k)
        self.edge3 = EdgeConv(64, local_dim, k)
        self.fuse = nn.Sequential(
            nn.Conv1d(64 + 64 + local_dim, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.identity_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )
        self.quality_head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        # points: [B, N, C]
        x = points.transpose(2, 1).contiguous()
        x1 = self.edge1(x)
        x2 = self.edge2(x1)
        local = self.edge3(x2)
        fused = self.fuse(torch.cat([x1, x2, local], dim=1))
        global_max = F.adaptive_max_pool1d(fused, 1).squeeze(-1)
        global_avg = F.adaptive_avg_pool1d(fused, 1).squeeze(-1)
        global_feature = torch.cat([global_max, global_avg], dim=1)
        identity = F.normalize(self.identity_head(global_feature), dim=1)
        quality = self.quality_head(global_feature).squeeze(1)
        return {
            "identity": identity,
            "local": local.transpose(2, 1).contiguous(),
            "global": global_feature,
            "quality": quality,
        }


class BuildingInvariantEncoder(nn.Module):
    """Lightweight identity branch built only from rigid-motion invariants."""

    def __init__(
        self,
        embedding_dim: int = 256,
        local_dim: int = 128,
        k: int = 20,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.k = int(k)
        self.local_net = nn.Sequential(
            nn.Conv1d(11, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, local_dim, 1, bias=False),
            nn.BatchNorm1d(local_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(local_dim, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.identity_head = nn.Sequential(
            nn.Linear(512 + 9, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

    def invariant_features(self, points: torch.Tensor) -> torch.Tensor:
        xyz, normals = points[..., :3], F.normalize(points[..., 3:6], dim=-1)
        centered = xyz - xyz.mean(dim=1, keepdim=True)
        scale = torch.linalg.vector_norm(centered, dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-8)
        normalized = centered / scale[..., None]
        distances = torch.cdist(normalized, normalized)
        effective_k = min(self.k + 1, points.shape[1])
        values, indices = distances.topk(effective_k, largest=False)
        values, indices = values[..., 1:], indices[..., 1:]
        if values.shape[-1] == 0:
            values, indices = distances.topk(1, largest=False)
        batch = torch.arange(points.shape[0], device=points.device)[:, None, None]
        neighbours = normalized[batch, indices]
        delta = neighbours - normalized[:, :, None, :]
        covariance = delta.transpose(-1, -2) @ delta / max(values.shape[-1], 1)
        covariance_invariants = self.covariance_invariants(covariance)
        radius = torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
        distance_stats = torch.stack(
            [values.mean(dim=-1), values.std(dim=-1, unbiased=False), values.amax(dim=-1), values.median(dim=-1).values],
            dim=-1,
        )
        radial = normalized / radius.clamp_min(1e-8)
        radial_alignment = torch.abs((normals * radial).sum(dim=-1, keepdim=True))
        neighbour_normals = normals[batch, indices]
        agreement = torch.abs((neighbour_normals * normals[:, :, None, :]).sum(dim=-1))
        normal_stats = torch.stack(
            [agreement.mean(dim=-1), agreement.std(dim=-1, unbiased=False)], dim=-1
        )
        local = torch.cat(
            [radius, distance_stats, covariance_invariants, radial_alignment, normal_stats], dim=-1
        )
        position_covariance = normalized.transpose(1, 2) @ normalized / max(points.shape[1] - 1, 1)
        position_invariants = self.covariance_invariants(position_covariance)
        radial_quantiles = torch.quantile(
            radius.squeeze(-1),
            torch.tensor([0.25, 0.5, 0.75, 0.95], device=points.device),
            dim=1,
        ).transpose(0, 1)
        radial_quantiles = radial_quantiles / radial_quantiles[:, -1:].clamp_min(1e-8)
        normal_second_moment = normals.transpose(1, 2) @ normals / max(points.shape[1], 1)
        normal_invariants = self.covariance_invariants(normal_second_moment)[:, 1:]
        global_features = torch.cat(
            [position_invariants, radial_quantiles, normal_invariants], dim=1
        )
        return local, global_features

    @staticmethod
    def covariance_invariants(covariance: torch.Tensor) -> torch.Tensor:
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-8)
        frobenius = torch.sqrt(torch.sum(covariance * covariance, dim=(-2, -1)).clamp_min(0.0))
        a, b, c = covariance[..., 0, 0], covariance[..., 0, 1], covariance[..., 0, 2]
        d, e, f = covariance[..., 1, 0], covariance[..., 1, 1], covariance[..., 1, 2]
        g, h, i = covariance[..., 2, 0], covariance[..., 2, 1], covariance[..., 2, 2]
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        return torch.stack(
            [trace, frobenius / trace, determinant / trace.pow(3)], dim=-1
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        local_invariant, global_invariant = self.invariant_features(points)
        encoded = self.local_net(local_invariant.transpose(1, 2).contiguous())
        global_max = F.adaptive_max_pool1d(encoded, 1).squeeze(-1)
        global_avg = F.adaptive_avg_pool1d(encoded, 1).squeeze(-1)
        global_feature = torch.cat([global_max, global_avg, global_invariant], dim=1)
        identity = F.normalize(self.identity_head(global_feature), dim=1)
        return {
            "identity": identity,
            "local": encoded.transpose(1, 2).contiguous(),
            "global": global_feature,
            "invariant": local_invariant,
        }


class FaceGraphConv(nn.Module):
    """Message passing over the three true shared-edge neighbors of each face."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim, bias=False),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(x.shape[0], device=x.device)[:, None, None]
        neighbor_features = x[batch, neighbors]
        centers = x[:, :, None, :].expand_as(neighbor_features)
        return self.net(torch.cat([neighbor_features - centers, centers], dim=-1)).max(dim=2).values


class MeshFaceEncoder(nn.Module):
    """Native triangle-face graph encoder with an explicit topology summary."""

    def __init__(
        self,
        input_dim: int = 13,
        topology_dim: int = 8,
        embedding_dim: int = 256,
        local_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.face1 = FaceGraphConv(input_dim, 64)
        self.face2 = FaceGraphConv(64, 64)
        self.face3 = FaceGraphConv(64, local_dim)
        self.fuse = nn.Sequential(
            nn.Linear(64 + 64 + local_dim, 512, bias=False),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.identity_head = nn.Sequential(
            nn.Linear(1024 + topology_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

    def forward(
        self,
        face_features: torch.Tensor,
        neighbors: torch.Tensor,
        mask: torch.Tensor,
        topology: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x1 = self.face1(face_features, neighbors)
        x2 = self.face2(x1, neighbors)
        local = self.face3(x2, neighbors)
        fused = self.fuse(torch.cat([x1, x2, local], dim=-1))
        valid = mask[:, :, None]
        global_max = fused.masked_fill(~valid, torch.finfo(fused.dtype).min).max(dim=1).values
        global_avg = (fused * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        global_feature = torch.cat([global_max, global_avg, topology], dim=1)
        identity = F.normalize(self.identity_head(global_feature), dim=1)
        return {
            "identity": identity,
            "local": local,
            "face": fused,
            "global": global_feature,
        }


class QualityComparator(nn.Module):
    """Full-reference downstream head comparing clean and distorted Base features."""

    def __init__(self, feature_dim: int = 1024, dropout: float = 0.2):
        super().__init__()
        comparison_dim = feature_dim * 2 + 1
        self.net = nn.Sequential(
            nn.LayerNorm(comparison_dim),
            nn.Linear(comparison_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, reference: torch.Tensor, distorted: torch.Tensor) -> torch.Tensor:
        cosine = F.cosine_similarity(reference, distorted, dim=1).unsqueeze(1)
        comparison = torch.cat(
            [torch.abs(reference - distorted), reference * distorted, cosine], dim=1
        )
        return self.net(comparison).squeeze(1)


class QualityVectorComparator(nn.Module):
    """Predict multiple standardized objective quality dimensions from Base feature pairs."""

    def __init__(self, feature_dim: int = 1024, output_dim: int = 6, dropout: float = 0.2):
        super().__init__()
        comparison_dim = feature_dim * 2 + 1
        self.net = nn.Sequential(
            nn.LayerNorm(comparison_dim),
            nn.Linear(comparison_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_dim),
        )

    def forward(self, reference: torch.Tensor, degraded: torch.Tensor) -> torch.Tensor:
        cosine = F.cosine_similarity(reference, degraded, dim=1).unsqueeze(1)
        comparison = torch.cat(
            [torch.abs(reference - degraded), reference * degraded, cosine], dim=1
        )
        return self.net(comparison)


def symmetric_nt_xent(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = a @ b.T / temperature
    targets = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def local_correspondence_loss(
    reference: torch.Tensor,
    transformed: torch.Tensor,
    temperature: float = 0.1,
    samples: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contrast corresponding point features against other points of the same shape."""
    point_count = reference.shape[1]
    selected = torch.randperm(point_count, device=reference.device)[: min(samples, point_count)]
    a = F.normalize(reference[:, selected], dim=2)
    b = F.normalize(transformed[:, selected], dim=2)
    logits = torch.matmul(a, b.transpose(2, 1)) / temperature
    targets = torch.arange(len(selected), device=reference.device).expand(reference.shape[0], -1)
    loss = 0.5 * (
        F.cross_entropy(logits.reshape(-1, len(selected)), targets.reshape(-1))
        + F.cross_entropy(logits.transpose(2, 1).reshape(-1, len(selected)), targets.reshape(-1))
    )
    accuracy = (logits.argmax(dim=2) == targets).float().mean()
    return loss, accuracy


def patch_correspondence_loss(
    reference_features: torch.Tensor,
    transformed_features: torch.Tensor,
    reference_xyz: torch.Tensor,
    temperature: float = 0.1,
    anchors: int = 16,
    neighbors: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-building InfoNCE on pooled local neighborhoods with known correspondence."""
    batch, point_count, _ = reference_features.shape
    anchor_indices = torch.randperm(point_count, device=reference_features.device)[: min(anchors, point_count)]
    anchor_xyz = reference_xyz[:, anchor_indices, :3]
    distances = torch.cdist(anchor_xyz, reference_xyz[:, :, :3])
    neighbor_indices = distances.topk(min(neighbors, point_count), largest=False, dim=2).indices
    batch_indices = torch.arange(batch, device=reference_features.device)[:, None, None]
    reference_patches = reference_features[batch_indices, neighbor_indices].mean(dim=2)
    transformed_patches = transformed_features[batch_indices, neighbor_indices].mean(dim=2)
    reference_patches = F.normalize(reference_patches.reshape(-1, reference_features.shape[-1]), dim=1)
    transformed_patches = F.normalize(transformed_patches.reshape(-1, transformed_features.shape[-1]), dim=1)
    logits = reference_patches @ transformed_patches.T / temperature
    targets = torch.arange(len(reference_patches), device=reference_features.device)
    loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return loss, accuracy
