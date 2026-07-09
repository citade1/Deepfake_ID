"""LID estimator.

LID is measured against a fixed real-image reference bank, so a sample's
estimate depends only on itself and single-image inference is valid. See README.
"""

import torch


def twonn_global_id(features: torch.Tensor, discard_frac: float = 0.1) -> float:
    """Global ID via TwoNN (Facco et al., 2017): slope of -log(1-F(mu)) vs
    log(mu), mu=r2/r1, dropping the noisy top discard_frac. For layer profiling."""
    N = features.size(0)
    if N < 10:
        return float("nan")
    d = torch.cdist(features, features, p=2)
    d.fill_diagonal_(float("inf"))
    two, _ = torch.topk(d, 2, dim=1, largest=False)          # r1, r2
    mu, _ = torch.sort((two[:, 1] / (two[:, 0] + 1e-12)).clamp(min=1 + 1e-9))
    keep = int(N * (1 - discard_frac))
    mu = mu[:keep]
    cdf = torch.arange(1, keep + 1, device=mu.device, dtype=mu.dtype) / N
    x, y = torch.log(mu), -torch.log(1 - cdf)
    return float((x * y).sum() / (x * x).sum())              # slope through origin


def compute_lid_features(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int = 20,
) -> torch.Tensor:
    """Per-query k-NN log-ratio LID features against a fixed reference set.

    Returns (Q, k), zero-padded if reference has fewer than k points. 
    """
    R, Q, eps = reference.size(0), query.size(0), 1e-9
    dist = torch.cdist(query, reference, p=2)

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
