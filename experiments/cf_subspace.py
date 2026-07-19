"""How many fakeness directions suffice to generalize? Compare, per held-out family,
a probe in the training-family direction subspace vs full 768-d vs 1-D. See README."""
import argparse
import random
import statistics as st

import numpy as np
import torch

from utils.cf_data import GENERATIONS, load
from utils.figs import plt, save_fig, save_json
from utils.heads import auc, fit_head

CAP_TR = 700


def family_dirs(X, y, fam, families, mu_real):
    cols = []
    for g in families:
        m = torch.tensor([fam[i] == g and y[i] == 1 for i in range(len(y))])
        v = X[m].mean(0) - mu_real
        cols.append(v / v.norm())
    Q, _ = torch.linalg.qr(torch.stack(cols).T)          # (768, k) orthonormal basis
    return Q


def fit_auc(X, y, tr, va, te):
    tr, va, te = torch.tensor(tr), torch.tensor(va), torch.tensor(te)
    return auc(fit_head(X[tr], y[tr], X[va], y[va], seed=0), X[te], y[te])


def run(seed, backbone):
    rng = random.Random(seed)
    d = load(seed, backbone)
    y, fam, X = d["label"], d["family"], d["feat12"]
    real = (y == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real)
    real_tr, real_te = real[:CAP_TR], real[CAP_TR:CAP_TR + 2000]
    mu_real = X[torch.tensor(real_tr)].mean(0)
    fk = {g: [i for i in range(len(y)) if y[i] == 1 and fam[i] == g] for g in GENERATIONS}

    out = {}
    for G in GENERATIONS:
        others = [g for g in GENERATIONS if g != G]
        Q = family_dirs(X, y, fam, others, mu_real)      # basis from training families only
        shared = Q.mean(1, keepdim=True)
        shared = shared / shared.norm()
        other_fake = [i for g in others for i in fk[g]]
        rng.shuffle(other_fake)
        n = len(real_tr)
        tr_f, nvr = other_fake[:n], max(1, n // 10)
        tr, va, te = real_tr[nvr:] + tr_f[nvr:], real_tr[:nvr] + tr_f[:nvr], real_te + fk[G]
        out[G] = (fit_auc(X, y, tr, va, te), fit_auc(X @ Q, y, tr, va, te),
                  fit_auc(X @ shared, y, tr, va, te))
    return out


def main():
    ap = argparse.ArgumentParser(description="Fakeness-subspace generalization (LOGO)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    runs = [run(s, args.backbone) for s in range(args.seeds)]

    cols = ["full_768d", "subspace_4d", "shared_1d"]
    data = {G: {cols[k]: [sum(r[G][k] for r in runs) / len(runs),
                          st.stdev([r[G][k] for r in runs]) if len(runs) > 1 else 0.0]
                for k in range(3)} for G in GENERATIONS}

    print(f"leave-one-generation-out AUC, mean +/- std over {args.seeds} seeds")
    print(f"{'held-out':>12} | {'full768':>13} {'subspace':>13} {'1Dshared':>13}")
    for G in GENERATIONS:
        print(f"{G:>12} | " + " ".join(f"{data[G][c][0]:.3f}±{data[G][c][1]:.3f}" for c in cols))

    save_json({"backbone": args.backbone, "families": GENERATIONS, "seeds": args.seeds,
               "auc": data}, f"cf_subspace_{args.backbone}")
    plot(data, cols, args.backbone)


def plot(data, cols, backbone):
    x = np.arange(len(GENERATIONS))
    labels = {"full_768d": "full 768-d", "subspace_4d": "4-D subspace", "shared_1d": "1-D shared axis"}
    colors = {"full_768d": "#5b8ff9", "subspace_4d": "#61ddaa", "shared_1d": "#d1495b"}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, c in enumerate(cols):
        ax.bar(x + (k - 1) * 0.27, [data[G][c][0] for G in GENERATIONS], 0.27,
               yerr=[data[G][c][1] for G in GENERATIONS], label=labels[c], color=colors[c])
    ax.set_xticks(list(x))
    ax.set_xticklabels(GENERATIONS, rotation=45, ha="right")
    ax.set_ylabel("held-out AUC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    ax.set_title(f"Fakeness-direction generalization, LOGO ({backbone})", fontsize=10)
    save_fig(fig, f"cf_subspace_{backbone}")


if __name__ == "__main__":
    main()
