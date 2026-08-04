"""Train-family x test-family generalization matrix; each cell is a held-out AUC."""
import argparse
import random
import statistics as st

import numpy as np
import torch

from utils.cf_data import (GENERATIONS, balanced, fake_index, halves, load,
                           real_split, train_val)
from utils.figs import plt, save_fig, save_json
from utils.heads import auc, fit_head


def per_seed(seed, backbone):
    rng = random.Random(seed)
    d = load(seed, backbone)
    y, fam, X = d["label"], d["family"], d["feat12"]
    real = (y == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real)
    real_tr, real_te = real_split(real)
    sp = {g: halves(v) for g, v in fake_index(y, fam, rng).items()}
    fams = [g for g, v in sp.items() if v]
    ftr = {g: sp[g][0] for g in fams}
    fte = {g: sp[g][1] for g in fams}

    M = {}
    for gi in fams:                              # train family
        tr, va = train_val(balanced(real_tr, ftr[gi], rng), rng)
        mlp = fit_head(X[torch.tensor(tr)], y[torch.tensor(tr)],
                       X[torch.tensor(va)], y[torch.tensor(va)], seed=0)
        for gj in fams:                          # test family
            te = torch.tensor(real_te + fte[gj])
            M[f"{gi}|{gj}"] = auc(mlp, X[te], y[te])
    return M, fams


def heatmap(mean, fams, backbone):
    n = len(fams)
    A = np.array([[mean[f"{a}|{b}"] for b in fams] for a in fams])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(A, cmap="viridis", vmin=min(0.7, A.min()), vmax=1.0)  # never clip a low cell
    ax.set_xticks(range(n))
    ax.set_xticklabels(fams, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(fams)
    ax.set_xlabel("test family")
    ax.set_ylabel("train family")
    lo, hi = A.min(), 1.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    color="white" if A[i, j] < lo + 0.6 * (hi - lo) else "black", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, label="test AUC")
    ax.set_title(f"Cross-architecture transfer ({backbone})", fontsize=10)
    save_fig(fig, f"cf_matrix_{backbone}")


def main():
    ap = argparse.ArgumentParser(description="Architecture generalization matrix")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    runs, famsets = zip(*[per_seed(s, args.backbone) for s in range(args.seeds)])
    fams = [g for g in GENERATIONS if all(g in f for f in famsets)]
    keys = [f"{a}|{b}" for a in fams for b in fams]
    mean = {k: sum(r[k] for r in runs) / len(runs) for k in keys}
    std = {k: (st.stdev([r[k] for r in runs]) if len(runs) > 1 else 0.0) for k in keys}

    w = max(len(g) for g in fams)
    print(f"rows = train, cols = test | mean AUC over {args.seeds} seeds\n")
    print(" " * (w + 2) + "  ".join(f"{g[:8]:>8}" for g in fams))
    for gi in fams:
        print(f"{gi:>{w}}  " + "  ".join(f"{mean[f'{gi}|{gj}']:8.3f}" for gj in fams))
    off = [mean[f"{a}|{b}"] for a in fams for b in fams if a != b]
    print(f"\nmean off-diagonal (cross-architecture) AUC: {sum(off)/len(off):.3f}")

    save_json({"backbone": args.backbone, "families": fams, "seeds": args.seeds,
               "mean": mean, "std": std}, f"cf_matrix_{args.backbone}")
    heatmap(mean, fams, args.backbone)


if __name__ == "__main__":
    main()
