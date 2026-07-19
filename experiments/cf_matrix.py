"""Train-family x test-family generalization matrix (raw CLS probe): each row trains
on real + one family's fakes, each column tests on another. Cell = AUC. See README."""
import argparse
import random
import statistics as st

import numpy as np
import torch

from utils.cf_data import GENERATIONS, load
from utils.figs import plt, save_fig, save_json
from utils.heads import auc, fit_head

CAP_TR, CAP_TE, REAL_TE = 700, 700, 2000


def per_seed(seed, backbone):
    rng = random.Random(seed)
    d = load(seed, backbone)
    y, fam, X = d["label"], d["family"], d["feat12"]
    real = (y == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real)
    real_tr, real_te = real[:CAP_TR], real[CAP_TR:CAP_TR + REAL_TE]
    ftr, fte = {}, {}
    for g in GENERATIONS:
        idx = [i for i in range(len(y)) if y[i] == 1 and fam[i] == g]
        rng.shuffle(idx)
        half = len(idx) // 2
        ftr[g], fte[g] = idx[:half][:CAP_TR], idx[half:][:CAP_TE]

    M = {}
    for gi in GENERATIONS:                       # train family
        pool = real_tr + ftr[gi]
        rng.shuffle(pool)
        nv = len(pool) // 10
        va, tr = torch.tensor(pool[:nv]), torch.tensor(pool[nv:])
        mlp = fit_head(X[tr], y[tr], X[va], y[va], seed=0)
        for gj in GENERATIONS:                   # test family
            te = torch.tensor(real_te + fte[gj])
            M[f"{gi}|{gj}"] = auc(mlp, X[te], y[te])
    return M


def heatmap(mean, backbone):
    n = len(GENERATIONS)
    A = np.array([[mean[f"{a}|{b}"] for b in GENERATIONS] for a in GENERATIONS])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(A, cmap="viridis", vmin=0.7, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels(GENERATIONS, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(GENERATIONS)
    ax.set_xlabel("test family")
    ax.set_ylabel("train family")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    color="white" if A[i, j] < 0.88 else "black", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, label="test AUC")
    ax.set_title(f"Cross-architecture transfer ({backbone})", fontsize=10)
    save_fig(fig, f"cf_matrix_{backbone}")


def main():
    ap = argparse.ArgumentParser(description="Architecture generalization matrix")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    runs = [per_seed(s, args.backbone) for s in range(args.seeds)]
    keys = runs[0].keys()
    mean = {k: sum(r[k] for r in runs) / len(runs) for k in keys}
    std = {k: (st.stdev([r[k] for r in runs]) if len(runs) > 1 else 0.0) for k in keys}

    w = max(len(g) for g in GENERATIONS)
    print(f"rows = train, cols = test | mean AUC over {args.seeds} seeds\n")
    print(" " * (w + 2) + "  ".join(f"{g[:8]:>8}" for g in GENERATIONS))
    for gi in GENERATIONS:
        print(f"{gi:>{w}}  " + "  ".join(f"{mean[f'{gi}|{gj}']:8.3f}" for gj in GENERATIONS))
    off = [mean[f"{a}|{b}"] for a in GENERATIONS for b in GENERATIONS if a != b]
    print(f"\nmean off-diagonal (cross-architecture) AUC: {sum(off)/len(off):.3f}")

    save_json({"backbone": args.backbone, "families": GENERATIONS, "seeds": args.seeds,
               "mean": mean, "std": std}, f"cf_matrix_{args.backbone}")
    heatmap(mean, args.backbone)


if __name__ == "__main__":
    main()
