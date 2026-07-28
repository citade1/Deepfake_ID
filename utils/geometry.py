"""Shared geometry helpers: the generative axis d = mu_fake - mu_real and the
whitened discriminative axis w = Sigma^-1 d (Fisher LDA)."""
import torch


def unit(v):
    return v / (v.norm() + 1e-12)


def shrink_cov(X, alpha=0.3):
    """Ledoit-Wolf-style shrunk covariance so a 768/1024-d Sigma stays invertible."""
    Xc = X - X.mean(0)
    S = (Xc.T @ Xc) / max(len(X) - 1, 1)
    d = S.shape[0]
    eye = torch.eye(d, dtype=S.dtype, device=S.device)      # match S so solve/inv never mismatch
    return (1 - alpha) * S + alpha * (torch.trace(S) / d) * eye


def axes(real, fake, alpha=0.3):
    """Return (w, d_hat): whitened discriminative axis Sigma^-1 (mu_fake-mu_real),
    and the unit generative axis mu_fake - mu_real."""
    d = fake.mean(0) - real.mean(0)
    S = 0.5 * (shrink_cov(real, alpha) + shrink_cov(fake, alpha))
    return unit(torch.linalg.solve(S, d)), unit(d)

