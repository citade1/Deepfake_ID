"""Held-out detection with one cleaned axis w vs the raw axis d vs a full MLP probe."""
import argparse
import random
import statistics as st

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils.cf_data import (FAKE_FIT, FAKE_TEST, GENERATIONS, balanced, fake_index, load,
                           real_split, train_val)
from utils.figs import plt, save_fig, save_json
from utils.geometry import axes
from utils.heads import auc as mlp_auc
from utils.heads import fit_head

ARMS = ["generative_d", "whitened_w", "full_mlp"]


def per_seed(seed, backbone):
    rng = random.Random(seed)
    d = load(seed, backbone)
    y, fam, X = d["label"], d["family"], d["feat12"]
    real = (y == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real)
    real_fit, real_te = real_split(real)
    fk = fake_index(y, fam, rng)

    out = {}
    for G in fk:                                          # leave family G out
        seen = [i for g, v in fk.items() if g != G for i in v[:FAKE_FIT]]
        pool = balanced(real_fit, seen, rng)              # the axes and the MLP get the same data
        pr = torch.tensor([i for i in pool if y[i] == 0])
        pf = torch.tensor([i for i in pool if y[i] == 1])
        w, dhat = axes(X[pr], X[pf])
        tr, va = train_val(pool, rng)
        mlp = fit_head(X[torch.tensor(tr)], y[torch.tensor(tr)],
                       X[torch.tensor(va)], y[torch.tensor(va)], seed=0)
        te = torch.tensor(real_te + fk[G][FAKE_FIT:FAKE_FIT + FAKE_TEST])
        yt = y[te]
        out[G] = (roc_auc_score(yt, X[te] @ dhat), roc_auc_score(yt, X[te] @ w),
                  mlp_auc(mlp, X[te], yt))
    return out, X.shape[1]


def main():
    ap = argparse.ArgumentParser(description="Whitening (LDA axis) ablation, leave-one-generation-out")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    R, dims = zip(*[per_seed(s, args.backbone) for s in range(args.seeds)])
    dim = dims[0]
    fams = [g for g in GENERATIONS if all(g in r for r in R)]

    def ms(g, k):
        v = [r[g][k] for r in R]
        return [sum(v) / len(v), st.stdev(v) if len(v) > 1 else 0.0]
    data = {g: {ARMS[k]: ms(g, k) for k in range(3)} for g in fams}

    print(f"held-out transfer AUC, mean +/- std over {args.seeds} seeds ({args.backbone})")
    print(f"{'held-out':>12} | {'raw d (1D)':>13} {'whitened w (1D)':>16} {'full MLP':>13}")
    for g in fams:
        print(f"{g:>12} | " + " ".join(f"{data[g][a][0]:.3f}±{data[g][a][1]:.3f}" for a in ARMS))

    save_json({"backbone": args.backbone, "seeds": args.seeds, "dim": dim, "auc": data},
              f"cf_whiten_{args.backbone}")
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for k, (key, lab, col) in enumerate([("generative_d", "raw axis d (1-D)", "#5b8ff9"),
                                         ("whitened_w", "whitened w = Σ⁻¹d (1-D)", "#d1495b"),
                                         ("full_mlp", f"full MLP ({dim}-d)", "#61ddaa")]):
        ax.bar(x + (k - 1) * 0.27, [data[g][key][0] for g in fams], 0.27,
               yerr=[data[g][key][1] for g in fams], label=lab, color=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels(fams, rotation=45, ha="right")
    ax.set_ylabel("held-out AUC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    ax.set_title(f"One whitened axis vs a full MLP, held-out generators ({args.backbone})", fontsize=10)
    save_fig(fig, f"cf_whiten_{args.backbone}")


if __name__ == "__main__":
    main()
