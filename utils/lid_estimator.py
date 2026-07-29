"""Intrinsic-dimension estimators: TwoNN (one number per set) and per-image LID."""
import torch


def twonn_global_id(features, discard_frac=0.1):
    """Set-level intrinsic dimension (TwoNN, Facco 2017), from 2nd/1st NN(Nearest Neighbor) distance ratios."""
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


def compute_lid_features(query, reference, k=20):
    """Per-image local complexity: log-ratio of k-NN distances to the reference set, (Q, k)."""
    R, Q, eps = reference.size(0), query.size(0), 1e-9
    k_eff = min(k, R)
    if k_eff < 1:
        return torch.zeros(Q, k, device=query.device, dtype=query.dtype)
    sorted_d, _ = torch.sort(torch.cdist(query, reference, p=2), dim=1)
    log_ratios = torch.log((sorted_d[:, k_eff - 1:k_eff] + eps) / (sorted_d[:, :k_eff] + eps))
    if k_eff < k:
        log_ratios = torch.cat([log_ratios, torch.zeros(Q, k - k_eff, device=query.device)], dim=1)
    return log_ratios
