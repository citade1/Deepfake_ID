"""Per-family probes ensembled (max/mean) vs one pooled probe, in-distribution."""
import argparse
import random
import statistics as st

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils.cf_data import (FAKE_FIT, FAKE_TEST, GENERATIONS, balanced, fake_index, load,
                           real_split, train_val)
from utils.figs import plt, save_fig, save_json
from utils.heads import fit_head, prob_fake

ARMS = ["pooled", "ens_max", "ens_mean"]


def train_probe(X, y, real_idx, fake_idx, rng):
    tr, va = train_val(balanced(real_idx, fake_idx, rng), rng)
    return fit_head(X[torch.tensor(tr)], y[torch.tensor(tr)],
                    X[torch.tensor(va)], y[torch.tensor(va)], seed=0)


def auc(y, p):
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return float("nan")


def per_seed(seed, backbone):
    rng = random.Random(seed)
    d = load(seed, backbone)
    y, fam, X = d["label"], d["family"], d["feat12"]
    real = (y == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real)
    real_fit, real_te = real_split(real)
    fk = fake_index(y, fam, rng)

    # each specialist is 1:1 on its own family; the pooled probe sees every family's fit half
    fam_probe = {g: train_probe(X, y, real_fit, v[:FAKE_FIT], rng) for g, v in fk.items()}
    pooled = train_probe(X, y, real_fit, [i for v in fk.values() for i in v[:FAKE_FIT]], rng)

    def scores(idx):
        idx = torch.tensor(idx)
        stack = torch.stack([prob_fake(fam_probe[g], X[idx]) for g in fk])
        return prob_fake(pooled, X[idx]), stack.max(0).values, stack.mean(0)

    fte = {g: v[FAKE_FIT:FAKE_FIT + FAKE_TEST] for g, v in fk.items()}
    res = {}
    for g in list(fk) + ["ALL"]:
        te = real_te + (fte[g] if g != "ALL" else [i for v in fte.values() for i in v])
        yt = y[torch.tensor(te)]
        res[g] = dict(zip(ARMS, (auc(yt, s) for s in scores(te))))
    return res


def main():
    ap = argparse.ArgumentParser(description="Ensemble vs pooled probe")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    R = [per_seed(s, args.backbone) for s in range(args.seeds)]
    cols = [g for g in GENERATIONS if all(g in r for r in R)] + ["ALL"]
    ms = lambda v: [sum(v) / len(v), st.stdev(v) if len(v) > 1 else 0.0]
    res = {g: {a: ms([r[g][a] for r in R]) for a in ARMS} for g in cols}

    print(f"test AUC, mean +/- std over {args.seeds} seeds ({args.backbone})")
    print(f"{'test set':>12} | " + " ".join(f"{a:>13}" for a in ARMS))
    for g in cols:
        print(f"{g:>12} | " + " ".join(f"{res[g][a][0]:.3f}±{res[g][a][1]:.3f}" for a in ARMS))

    save_json({"backbone": args.backbone, "seeds": args.seeds, "auc": res},
              f"cf_ensemble_{args.backbone}")
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for k, (key, lab, col) in enumerate([("pooled", "pooled probe", "#5b8ff9"),
                                         ("ens_max", "ensemble (max)", "#61ddaa"),
                                         ("ens_mean", "ensemble (mean)", "#d1495b")]):
        ax.bar(x + (k - 1) * 0.27, [res[g][key][0] for g in cols], 0.27,
               yerr=[res[g][key][1] for g in cols], label=lab, color=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_ylabel("test AUC")
    ax.set_ylim(0.9, 1.0)
    ax.legend(fontsize=8)
    ax.set_title(f"Ensemble vs pooled probe, in-distribution ({args.backbone})", fontsize=10)
    save_fig(fig, f"cf_ensemble_{args.backbone}")


if __name__ == "__main__":
    main()
