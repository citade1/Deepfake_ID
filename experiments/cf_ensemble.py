"""Experiment A: an ensemble of per-family probes (decision-level max/mean) vs one
pooled probe, in-distribution. See README / notes for the result."""
import argparse
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils.cf_data import GENERATIONS, load
from utils.figs import plt, save_fig, save_json
from utils.heads import fit_head, prob_fake

CAP = 700


def balanced(real, fakes, rng):
    rng.shuffle(fakes)
    n = min(len(real), len(fakes))
    return real[:n] + fakes[:n]


def train_probe(X, y, real_idx, fake_idx, rng, seed):
    idx = balanced(real_idx, fake_idx, rng)
    rng.shuffle(idx)
    nv = len(idx) // 10
    va, tr = torch.tensor(idx[:nv]), torch.tensor(idx[nv:])
    return fit_head(X[tr], y[tr], X[va], y[va], seed=seed)


def auc(y, p):
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description="Ensemble vs pooled probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    d = load(args.seed, args.backbone)
    y, fam, X = d["label"], d["family"], d["feat12"]

    real = (y == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real)
    real_tr, real_te = real[:3500], real[3500:5500]
    fk_tr, fk_te = {}, {}
    for g in GENERATIONS:
        idx = [i for i in range(len(y)) if y[i] == 1 and fam[i] == g]
        rng.shuffle(idx)
        h = len(idx) // 2
        fk_tr[g], fk_te[g] = idx[:h][:CAP], idx[h:]

    # per-family specialists + one pooled generalist
    fam_probe = {g: train_probe(X, y, list(real_tr), list(fk_tr[g]), rng, seed=0)
                 for g in GENERATIONS}
    pooled = train_probe(X, y, list(real_tr),
                         [i for g in GENERATIONS for i in fk_tr[g]], rng, seed=0)

    def scores(idx):
        idx = torch.tensor(idx)
        pooled_s = prob_fake(pooled, X[idx])
        stack = torch.stack([prob_fake(fam_probe[g], X[idx]) for g in GENERATIONS])
        return pooled_s, stack.max(0).values, stack.mean(0)   # pooled, ens-max, ens-mean

    res = {}
    print(f"{'test set':>12} | {'pooled':>7} {'ens-max':>7} {'ens-mean':>8}")
    for g in list(GENERATIONS) + ["ALL"]:
        te = real_te + (fk_te[g] if g != "ALL" else [i for gg in GENERATIONS for i in fk_te[gg]])
        yt = y[torch.tensor(te)]
        ps, mx, mn = scores(te)
        res[g] = {"pooled": auc(yt, ps), "ens_max": auc(yt, mx), "ens_mean": auc(yt, mn)}
        print(f"{g:>12} | {res[g]['pooled']:>7.3f} {res[g]['ens_max']:>7.3f} {res[g]['ens_mean']:>8.3f}")

    save_json({"backbone": args.backbone, "seed": args.seed, "auc": res}, f"cf_ensemble_{args.backbone}")
    fams = list(GENERATIONS) + ["ALL"]
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for k, (key, lab, col) in enumerate([("pooled", "pooled probe", "#5b8ff9"),
                                         ("ens_max", "ensemble (max)", "#61ddaa"),
                                         ("ens_mean", "ensemble (mean)", "#d1495b")]):
        ax.bar(x + (k - 1) * 0.27, [res[g][key] for g in fams], 0.27, label=lab, color=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels(fams, rotation=45, ha="right")
    ax.set_ylabel("test AUC")
    ax.set_ylim(0.9, 1.0)
    ax.legend(fontsize=8)
    ax.set_title(f"Ensemble vs pooled probe, in-distribution ({args.backbone})", fontsize=10)
    save_fig(fig, f"cf_ensemble_{args.backbone}")


if __name__ == "__main__":
    main()
