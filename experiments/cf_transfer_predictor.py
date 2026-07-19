"""Transfer-failure predictor: does the label-free geometric alignment
cos(d_B, w_A) -- how the held-out family B's shift lines up with the discriminative
axis a detector trained on A actually reads -- predict the real A->B transfer AUC?
A positive correlation turns the geometry into a label-free 'will it generalize?' test."""
import argparse
import random

import numpy as np
import torch

from utils.cf_data import GENERATIONS, load
from utils.figs import plt, save_fig, save_json
from utils.geometry import axes
from utils.heads import auc, fit_head

CAP_TR, CAP_TE, REAL_TE = 600, 600, 1500


def per_seed(seed, backbone):
    rng = random.Random(seed)
    d = load(seed, backbone)
    y, fam, X = d["label"], d["family"], d["feat12"]
    real = (y == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real)
    real_tr, real_te = real[:CAP_TR], real[CAP_TR:CAP_TR + REAL_TE]
    Xr_tr = X[torch.tensor(real_tr)]
    ftr, fte = {}, {}
    for g in GENERATIONS:
        idx = [i for i in range(len(y)) if y[i] == 1 and fam[i] == g]
        rng.shuffle(idx)
        h = len(idx) // 2
        ftr[g], fte[g] = idx[:h][:CAP_TR], idx[h:][:CAP_TE]

    # per train-family A: a detector (MLP head), the discriminative axis w_A, the generative d_A
    heads, wA, dA = {}, {}, {}
    for A in GENERATIONS:
        pool = real_tr + ftr[A]
        rng.shuffle(pool)
        nv = len(pool) // 10
        va, tr = torch.tensor(pool[:nv]), torch.tensor(pool[nv:])
        heads[A] = fit_head(X[tr], y[tr], X[va], y[va], seed=0)
        wA[A], dA[A] = axes(Xr_tr, X[torch.tensor(ftr[A])])

    rows = []
    for A in GENERATIONS:
        for B in GENERATIONS:
            _, dB = axes(Xr_tr, X[torch.tensor(fte[B])])       # B's generative shift (label-free)
            gw = torch.dot(wA[A], dB).item()                   # discriminative-axis alignment
            gd = torch.dot(dA[A], dB).item()                   # generative-axis alignment (baseline)
            te = torch.tensor(real_te + fte[B])
            a = auc(heads[A], X[te], y[te])                    # actual A->B transfer AUC
            rows.append((A, B, gw, gd, a))
    return rows


def corr(xs, ys):
    x, y = np.array(xs), np.array(ys)
    pear = float(np.corrcoef(x, y)[0, 1])
    rx, ry = x.argsort().argsort(), y.argsort().argsort()     # ranks for Spearman
    spear = float(np.corrcoef(rx, ry)[0, 1])
    return pear, spear


def main():
    ap = argparse.ArgumentParser(description="Geometric predictor of cross-generator transfer AUC")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    rows = [r for s in range(args.seeds) for r in per_seed(s, args.backbone)]
    off = [(gw, gd, a) for A, B, gw, gd, a in rows if A != B]

    def report(sel):
        gw, gd, a = [r[0] for r in sel], [r[1] for r in sel], [r[2] for r in sel]
        return corr(gw, a), corr(gd, a)
    w_all, d_all = report([(gw, gd, a) for _, _, gw, gd, a in rows])
    w_off, d_off = report(off)

    print(f"transfer predictor ({args.backbone}, {args.seeds} seeds, {len(rows)} pairs)")
    print(f"  discriminative cos(d_B, w_A):  all P{w_all[0]:+.3f} S{w_all[1]:+.3f} | "
          f"off-diag P{w_off[0]:+.3f} S{w_off[1]:+.3f}")
    print(f"  generative     cos(d_B, d_A):  all P{d_all[0]:+.3f} S{d_all[1]:+.3f} | "
          f"off-diag P{d_off[0]:+.3f} S{d_off[1]:+.3f}   (baseline)")

    save_json({"backbone": args.backbone, "seeds": args.seeds,
               "discriminative": {"pearson_all": w_all[0], "spearman_all": w_all[1],
                                  "pearson_offdiag": w_off[0], "spearman_offdiag": w_off[1]},
               "generative_baseline": {"pearson_all": d_all[0], "spearman_all": d_all[1],
                                       "pearson_offdiag": d_off[0], "spearman_offdiag": d_off[1]},
               "pairs": [{"train": A, "test": B, "cos_dB_wA": gw, "cos_dB_dA": gd, "auc": a}
                         for A, B, gw, gd, a in rows]},
              f"cf_transfer_predictor_{args.backbone}")

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    diag = [(gw, a) for A, B, gw, gd, a in rows if A == B]
    ax.scatter([gw for gw, _, _ in off], [a for _, _, a in off], s=18, color="#5b8ff9",
               label="A→B (transfer)")
    ax.scatter([gw for gw, _ in diag], [a for _, a in diag], s=22, color="#d1495b",
               label="A→A (in-family)")
    ax.set_xlabel("geometric alignment  cos(d_B, w_A)")
    ax.set_ylabel("actual transfer AUC")
    ax.set_title(f"Geometry predicts transfer ({args.backbone})\n"
                 f"off-diag Spearman: w_A {w_off[1]:+.2f} vs d_A {d_off[1]:+.2f}", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    save_fig(fig, f"cf_transfer_predictor_{args.backbone}")


if __name__ == "__main__":
    main()
