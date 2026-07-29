"""Qualitative + failure analysis for three detectors (cleaned axis w, MLP probe,
Mahalanobis-to-real): per-family error rates, how much their errors OVERLAP, and image
grids of the fakes they all miss. The overlap number is the point; grids just illustrate."""
import argparse
import io
import random

import numpy as np
import torch

from PIL import Image

from utils import backbones as B
from utils.figs import plt, save_fig
from utils.geometry import axes, shrink_cov
from utils.heads import fit_head, prob_fake

POOL = "checkpoints/cf_cache/ft_pool_bal_pf700_r2400_f1200.pt"
FAMS = ["GAN", "PixelDiff", "LatentDiff", "Flow", "Commercial"]
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def feats(model, proc, pool, name, imgs, batch=32):
    out = [B.features(model, proc, pool, name, imgs[i:i + batch], DEV, layers=(12,))[12]
           for i in range(0, len(imgs), batch)]
    return torch.cat(out)


def grid(imgs, titles, name, ncol=6):
    n = len(imgs)
    if n == 0:
        return
    nrow = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2.2))
    axs = np.atleast_1d(axs).ravel()
    for ax, im, t in zip(axs, imgs, titles):
        ax.imshow(im); ax.set_title(t, fontsize=7); ax.axis("off")
    for ax in axs[n:]:
        ax.axis("off")
    save_fig(fig, name)


def scores_of(Xfit, yfit, X):
    """Three fakeness scores (higher = more fake), each fit on (Xfit, yfit)."""
    rf, ff = Xfit[yfit == 0], Xfit[yfit == 1]
    w, _ = axes(rf, ff)                                    # cleaned axis
    mu = rf.mean(0)
    sinv = torch.linalg.inv(shrink_cov(rf, 0.5))
    nv = len(Xfit) // 10
    mlp = fit_head(Xfit[nv:], yfit[nv:], Xfit[:nv], yfit[:nv], seed=0)
    diff = X - mu
    return {"w": X @ w,
            "MLP": prob_fake(mlp, X),
            "Maha": torch.einsum("ni,ij,nj->n", diff, sinv, diff)}


def main():
    ap = argparse.ArgumentParser(description="Three-detector failure + overlap analysis")
    ap.add_argument("--backbone", default="clip")
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = torch.load(POOL, weights_only=False)
    idx = list(range(len(pool)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[:args.sample]
    imgs = [Image.open(io.BytesIO(pool[i][0])).convert("RGB") for i in idx]
    fam = [pool[i][1] for i in idx]
    y = torch.tensor([pool[i][2] for i in idx])

    model, proc, poolm = B.load(args.backbone, DEV)
    X = feats(model, proc, poolm, args.backbone, imgs)

    h = len(X) // 2                                        # fit on first half, judge on second
    S = scores_of(X[:h], y[:h], X)
    te = list(range(h, len(X)))

    # per method: threshold at class-mean midpoint (fit half), then predicted "fake"
    miss = {}                                              # method -> set of test-fake indices scored real
    print(f"false-negative rate per family ({args.backbone}):")
    print(f"{'family':>12} | " + " ".join(f"{m:>6}" for m in S))
    for m, s in S.items():
        s = s.cpu()
        thr = 0.5 * (s[:h][y[:h] == 0].mean() + s[:h][y[:h] == 1].mean())
        miss[m] = {i for i in te if y[i] == 1 and s[i] < thr}   # fakes scored below threshold
    for g in FAMS:
        fk = [i for i in te if y[i] == 1 and fam[i] == g]
        if fk:
            row = " ".join(f"{sum(i in miss[m] for i in fk) / len(fk):6.2f}" for m in S)
            print(f"{g:>12} | {row}")

    # error overlap: do the methods miss the SAME fakes?
    print("\nmissed-fake overlap (Jaccard):")
    ms = list(S)
    for a in range(len(ms)):
        for b in range(a + 1, len(ms)):
            A, Bs = miss[ms[a]], miss[ms[b]]
            j = len(A & Bs) / max(len(A | Bs), 1)
            print(f"  {ms[a]} vs {ms[b]}: {j:.2f}  ({len(A & Bs)} shared / {len(A | Bs)} union)")
    allmiss = set.intersection(*miss.values()) if miss else set()
    anymiss = set.union(*miss.values()) if miss else set()
    print(f"  all three miss: {len(allmiss)} / {len(anymiss)} of every-method error")

    # illustrate: the fakes ALL three miss (robust hard cases, not method-specific)
    ids = sorted(allmiss, key=lambda i: S["w"][i].item())[:args.n]
    grid([imgs[i] for i in ids], [f"{fam[i]} | w{S['w'][i]:+.2f}" for i in ids],
         f"gallery_all3_miss_{args.backbone}")

    # score distribution for the headline method (w)
    s = S["w"].cpu()
    thr = 0.5 * (s[:h][y[:h] == 0].mean() + s[:h][y[:h] == 1].mean())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for g in ["REAL"] + FAMS:
        v = [s[i].item() for i in te if (fam[i] == g if g != "REAL" else y[i] == 0)]
        if v:
            ax.hist(v, bins=30, alpha=0.5, label=g, density=True)
    ax.axvline(thr.item(), color="k", ls="--", lw=1, label="threshold")
    ax.set_xlabel("cleaned-axis score  x·w"); ax.legend(fontsize=8)
    ax.set_title(f"Score distribution by family ({args.backbone})", fontsize=10)
    save_fig(fig, f"gallery_scores_{args.backbone}")


if __name__ == "__main__":
    main()
