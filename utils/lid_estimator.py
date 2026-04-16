"""
TwoNN-based Local Intrinsic Dimension (LID) estimators.

References:
    Facco et al. (2017) "Estimating the intrinsic dimension of datasets by a
    minimal neighborhood information" — TwoNN global ID.

    Amsaleg et al. (2015) "Estimating Local Intrinsic Dimensionality" — MLE
    k-NN estimator used for per-point LID feature vectors.
"""

import torch


def twonn_global_id(features: torch.Tensor) -> float:
    """
    Estimate global intrinsic dimensionality using TwoNN (Facco et al., 2017).

    For each point i: mu_i = r2_i / r1_i (ratio of 2nd to 1st NN distance).
    Under the Pareto model, the MLE of dimension d is:
        d = N / sum_i( log(mu_i) )

    Args:
        features: (N, D) float tensor — must have N >= 3.
    Returns:
        Scalar float — estimated intrinsic dimension.
    """
    N = features.size(0)
    if N < 3:
        return float("nan")

    dist = torch.cdist(features, features, p=2)   # (N, N)
    dist.fill_diagonal_(float("inf"))
    sorted_d, _ = torch.sort(dist, dim=1)          # (N, N)

    r1 = sorted_d[:, 0]                            # 1st NN distance
    r2 = sorted_d[:, 1]                            # 2nd NN distance
    mu = r2 / (r1 + 1e-9)
    mu = torch.clamp(mu, min=1.0 + 1e-9)           # Pareto requires mu >= 1

    global_id = float(N) / torch.log(mu).sum().item()
    return global_id


def compute_lid_features(features: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    Per-point LID feature vector via k-NN log-ratio (MLE hill estimator).

    For point i with k nearest-neighbor distances r_{i,1} <= ... <= r_{i,k}:
        feature_j = log( r_{i,k} / r_{i,j} ),  j = 1 .. k

    The scalar LID estimate is then: k / sum_j( feature_j ).

    This returns the full (B, k) vector so a downstream linear layer can
    learn to weight each distance ratio independently.

    Args:
        features: (B, D) float tensor.
        k:        number of nearest neighbors.  k < B required.
    Returns:
        (B, k) tensor of log-ratio features (non-negative).
        If B <= k, k is clamped to B-1; the remaining columns are zero-padded.
    """
    B = features.size(0)
    k_eff = min(k, B - 1)

    if k_eff < 1:
        # Cannot compute — return zero features
        return torch.zeros(B, k, device=features.device, dtype=features.dtype)

    dist = torch.cdist(features, features, p=2)    # (B, B)
    dist.fill_diagonal_(float("inf"))
    sorted_d, _ = torch.sort(dist, dim=1)          # (B, B)

    d_neighbors = sorted_d[:, :k_eff]              # (B, k_eff)
    d_k = sorted_d[:, k_eff - 1].unsqueeze(1)     # (B, 1)

    log_ratios = torch.log(d_k / (d_neighbors + 1e-9) + 1e-9)  # (B, k_eff)

    if k_eff < k:
        pad = torch.zeros(B, k - k_eff,
                          device=features.device, dtype=features.dtype)
        log_ratios = torch.cat([log_ratios, pad], dim=1)

    return log_ratios  # (B, k)


def compute_lid_scalar(features: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    Per-point scalar LID estimate: k / sum( log-ratio features ).

    Args:
        features: (B, D) float tensor.
        k:        number of nearest neighbors.
    Returns:
        (B,) float tensor of LID values.
    """
    log_ratios = compute_lid_features(features, k)           # (B, k)
    lid = k / (log_ratios.sum(dim=1) + 1e-9)                # (B,)
    return lid
