"""Stage 3a — spectral localization. Does the real/fake signal live in the
principal (semantic) or minor-axis (residual) subspace of the real reference
bank? (A) population spectra, (B) probes on top-k / bottom-k / random-k PCA
coordinates, (C) training-free residual-energy scores, in-dist + LOO. All from
cached features; PCA basis comes from the real bank only (no label leakage)."""
import argparse
import os
import random
import statistics as st

import torch
from sklearn.metrics import roc_auc_score

from experiments.multiseed import fit, test_auc
from utils.lid_estimator import compute_lid_features  # noqa: F401 (baseline ref)

S1 = "./checkpoints/stage1_cache"
POOL = "./checkpoints/loo_cache/pool.pt"
RAW_L, LID_L = 12, 7
KS_PROBE = [4, 16, 64, 256]
KS_SCORE = [16, 64, 256, 512]


def load(tag):
    b = torch.load(os.path.join(S1, f"{tag}.pt"), map_location="cpu", weights_only=False)
    return b["feats"], b["labels"]


def spectrum(X):
    lam = torch.linalg.eigvalsh(torch.cov((X - X.mean(0)).T)).clamp(min=0).flip(0)
    eff = ((lam.sum() ** 2) / (lam ** 2).sum()).item()
    top10 = (lam[:10].sum() / lam.sum()).item()
    return eff, top10


def pca(bank):
    mu = bank.mean(0)
    _, S, Vh = torch.linalg.svd(bank - mu, full_matrices=False)
    lam = (S ** 2) / (len(bank) - 1)
    return mu, Vh, lam


def auc(y, s):
    return roc_auc_score(y, s)


def main():
    ap = argparse.ArgumentParser(description="Stage 3a spectral localization")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    ftr, ytr = load("train")
    fva, yva = load("val")
    fte, yte = load("test")
    bank = load("ref")[0][RAW_L]
    mu, Vh, lam = pca(bank)

    print("=== A. population spectra (train split, cosine features) ===")
    for L in (LID_L, RAW_L):
        er, e10r = spectrum(ftr[L][ytr == 0])
        ef, e10f = spectrum(ftr[L][ytr == 1])
        print(f"layer {L:>2}: eff-rank real {er:6.1f} fake {ef:6.1f} | "
              f"top-10 energy real {e10r:.3f} fake {e10f:.3f}")

    def coords(X):
        return (X - mu) @ Vh.T          # columns ordered by bank variance

    Ztr, Zva, Zte = coords(ftr[RAW_L]), coords(fva[RAW_L]), coords(fte[RAW_L])

    print(f"\n=== B. probe on subspace coordinates (test AUC, {args.seeds} seeds) ===")
    print(f"{'k':>4} | {'top-k':>15} | {'bottom-k':>15} | {'random-k':>15}")
    for k in KS_PROBE:
        row = []
        for tag in ("top", "bottom", "random"):
            if tag == "top":
                sl = lambda Z: Z[:, :k]
            elif tag == "bottom":
                sl = lambda Z: Z[:, -k:]
            else:
                torch.manual_seed(7)
                Q, _ = torch.linalg.qr(torch.randn(Ztr.shape[1], k))
                sl = lambda Z: Z @ Q
            vals = [test_auc(fit(sl(Ztr), ytr, sl(Zva), yva, sd), sl(Zte), yte)
                    for sd in range(args.seeds)]
            row.append(f"{sum(vals)/len(vals):.4f}±{st.stdev(vals):.4f}")
        print(f"{k:>4} | {row[0]:>15} | {row[1]:>15} | {row[2]:>15}")
    print("(reference: full 768-d raw probe 0.9733, LID-only 0.6257)")

    print("\n=== C. training-free residual-energy scores ===")
    tot = (Zte ** 2).sum(1)
    for k in KS_SCORE:
        r = 1 - (Zte[:, :k] ** 2).sum(1) / tot                     # residual energy
        m = ((Zte[:, k:] ** 2) / lam[k:]).sum(1)                   # whitened minor axes
        print(f"k={k:>3}: resid-energy AUC {auc(yte, r):.4f} | minor-Mahalanobis {auc(yte, m):.4f}")

    print("\n--- LOO (training-free, per held-out generator) ---")
    blob = torch.load(POOL, map_location="cpu", weights_only=False)
    pf, gen, py = blob["feats"], blob["gen"], blob["y"]
    rng = random.Random(42)
    real_pos = (py == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real_pos)
    ref_pos, rest = real_pos[:1000], real_pos[1000:]
    te_real = rest[len(rest) // 2:]
    mu2, Vh2, lam2 = pca(pf[RAW_L][ref_pos])
    Zp = (pf[RAW_L] - mu2) @ Vh2.T
    totp = (Zp ** 2).sum(1)
    for k in (64, 256, 512):
        r_auc, m_auc = [], []
        for g in sorted({v for v in gen.tolist() if v != 0}):
            idx = torch.tensor(te_real + ((py == 1) & (gen == g)).nonzero(as_tuple=True)[0].tolist())
            r = 1 - (Zp[idx, :k] ** 2).sum(1) / totp[idx]
            m = ((Zp[idx, k:] ** 2) / lam2[k:]).sum(1)
            r_auc.append(auc(py[idx], r))
            m_auc.append(auc(py[idx], m))
        print(f"k={k:>3}: resid-energy mean {sum(r_auc)/len(r_auc):.4f} | "
              f"minor-Mahalanobis mean {sum(m_auc)/len(m_auc):.4f}")
    print("(reference: LID-only LOO mean 0.6058; raw probe LOO 0.9056)")


if __name__ == "__main__":
    main()
