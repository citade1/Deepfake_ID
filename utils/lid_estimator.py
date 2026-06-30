"""LID estimators.

LID is measured against a fixed real-image reference bank, so a sample's
estimate depends only on itself and single-image inference is valid. See README.
"""

import torch


def twonn_global_id(features: torch.Tensor) -> float:
    # Diagnostic global ID (Facco et al., 2017) — training logs only.
    N = features.size(0)
    if N < 3:
        return float("nan")
    dist = torch.cdist(features, features, p=2)
    dist.fill_diagonal_(float("inf"))
    sorted_d, _ = torch.sort(dist, dim=1)
    mu = torch.clamp(sorted_d[:, 1] / (sorted_d[:, 0] + 1e-9), min=1.0 + 1e-9)
    return float(N) / torch.log(mu).sum().item()


def compute_lid_features(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int = 20,
    exclude_self: bool = False,
) -> torch.Tensor:
    """Per-query k-NN log-ratio LID features against a fixed reference set.

    Returns (Q, k), zero-padded if reference has fewer than k points.
    exclude_self drops each query's distance-0 self match (only needed when
    query rows are also in the reference; disjoint pools make it unnecessary).
    """
    R, Q, eps = reference.size(0), query.size(0), 1e-9
    dist = torch.cdist(query, reference, p=2)

    if exclude_self:
        dist.scatter_(1, dist.argmin(dim=1, keepdim=True), float("inf"))
        R -= 1

    k_eff = min(k, R)
    if k_eff < 1:
        return torch.zeros(Q, k, device=query.device, dtype=query.dtype)

    sorted_d, _ = torch.sort(dist, dim=1)
    d_neighbors = sorted_d[:, :k_eff]
    d_k = sorted_d[:, k_eff - 1].unsqueeze(1)
    log_ratios = torch.log((d_k + eps) / (d_neighbors + eps))

    if k_eff < k:
        pad = torch.zeros(Q, k - k_eff, device=query.device, dtype=query.dtype)
        log_ratios = torch.cat([log_ratios, pad], dim=1)
    return log_ratios
