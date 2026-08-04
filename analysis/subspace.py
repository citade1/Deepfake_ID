"""How many fakeness directions generalize: subspace probe vs full feature vs 1-D."""
import argparse
import random
import statistics as st

import numpy as np
import torch

from utils.cf_data import (FAKE_FIT, FAKE_TEST, GENERATIONS, balanced, fake_index,
                           load, real_split, train_val)
from utils.figs import plt, save_fig, save_json
from utils.heads import auc, fit_head


def family_dirs(X, by_family, mu_real):
    cols = []
    for v in by_family.values():
        u = X[torch.tensor(v)].mean(0) - mu_real
        cols.append(u / u.norm())
    D = torch.stack(cols)                                # (k, d) unit family directions
    Q, _ = torch.linalg.qr(D.T)                          # (d, k) orthonormal basis
    shared = D.mean(0)                                   # average the directions, not Q:
    return Q, shared / shared.norm()                     # QR rotates and flips its columns


def fit_auc(X, y, tr, va, te):
    tr, va, te = torch.tensor(tr), torch.tensor(va), torch.tensor(te)
    return auc(fit_head(X[tr], y[tr], X[va], y[va], seed=0), X[te], y[te])


def run(seed, backbone):
    rng = random.Random(seed)
    d = load(seed, backbone)
    y, fam, X = d["label"], d["family"], d["feat12"]
    real = (y == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real)
    real_tr, real_te = real_split(real)
    mu_real = X[torch.tensor(real_tr)].mean(0)
    fk = fake_index(y, fam, rng)

    out, dims = {}, {}
    for G in fk:
        others = {g: v[:FAKE_FIT] for g, v in fk.items() if g != G}
        Q, shared = family_dirs(X, others, mu_real)           # from training families only
        shared = shared.unsqueeze(1)
        seen = [i for v in others.values() for i in v]
        tr, va = train_val(balanced(real_tr, seen, rng), rng)
        te = real_te + fk[G][FAKE_FIT:FAKE_FIT + FAKE_TEST]
        out[G] = (fit_auc(X, y, tr, va, te), fit_auc(X @ Q, y, tr, va, te),
                  fit_auc(X @ shared, y, tr, va, te))
        dims = {"full": X.shape[1], "subspace": Q.shape[1], "shared_1d": 1}
    return out, dims


def main():
    ap = argparse.ArgumentParser(description="Fakeness-subspace generalization (LOGO)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    runs, dims = zip(*[run(s, args.backbone) for s in range(args.seeds)])
    fams = [g for g in GENERATIONS if all(g in r for r in runs)]
    dim = dims[0]

    cols = ["full", "subspace", "shared_1d"]
    data = {G: {cols[k]: [sum(r[G][k] for r in runs) / len(runs),
                          st.stdev([r[G][k] for r in runs]) if len(runs) > 1 else 0.0]
                for k in range(3)} for G in fams}

    print(f"leave-one-generation-out AUC, mean +/- std over {args.seeds} seeds")
    head = [f"full {dim['full']}-d", f"subspace {dim['subspace']}-d", "shared 1-d"]
    print(f"{'held-out':>12} | " + " ".join(f"{h:>13}" for h in head))
    for G in fams:
        print(f"{G:>12} | " + " ".join(f"{data[G][c][0]:.3f}±{data[G][c][1]:.3f}" for c in cols))

    save_json({"backbone": args.backbone, "families": fams, "seeds": args.seeds,
               "dims": dim, "auc": data}, f"cf_subspace_{args.backbone}")
    plot(data, cols, fams, dim, args.backbone)


def plot(data, cols, fams, dim, backbone):
    x = np.arange(len(fams))
    labels = {"full": f"full {dim['full']}-d", "subspace": f"{dim['subspace']}-D subspace",
              "shared_1d": "1-D shared axis"}
    colors = {"full": "#5b8ff9", "subspace": "#61ddaa", "shared_1d": "#d1495b"}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, c in enumerate(cols):
        ax.bar(x + (k - 1) * 0.27, [data[G][c][0] for G in fams], 0.27,
               yerr=[data[G][c][1] for G in fams], label=labels[c], color=colors[c])
    ax.set_xticks(list(x))
    ax.set_xticklabels(fams, rotation=45, ha="right")
    ax.set_ylabel("held-out AUC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    ax.set_title(f"Fakeness-direction generalization, LOGO ({backbone})", fontsize=10)
    save_fig(fig, f"cf_subspace_{backbone}")


if __name__ == "__main__":
    main()
