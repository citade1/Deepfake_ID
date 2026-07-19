"""In-distribution difficulty and leave-one-generation-out generalization on CF."""
import argparse
import random

import numpy as np
import torch

from utils.cf_data import GENERATIONS, load
from utils.figs import plt, save_fig, save_json
from utils.heads import auc, fit_head
from utils.lid_estimator import compute_lid_features

REF_N, K = 1000, 20


def arms(d, ref_idx):
    ref = d["feat7"][ref_idx]
    raw = d["feat12"]
    lid = compute_lid_features(d["feat7"], ref, k=K)
    return {"raw": raw, "LID": lid, "raw+LID": torch.cat([raw, lid], dim=1)}


def evaluate(A, y, tr, va, te):
    tr, va, te = torch.tensor(tr), torch.tensor(va), torch.tensor(te)
    out = {}
    for name, X in A.items():
        mlp = fit_head(X[tr], y[tr], X[va], y[va], seed=0)
        out[name] = auc(mlp, X[te], y[te])
    return out


def balanced_train(tr_real, fake_pool, rng):
    """Train/val index lists with real:fake = 1:1 (subsample the fake majority)."""
    rng.shuffle(fake_pool)
    n = len(tr_real)
    tr_f, nvr = fake_pool[:n], max(1, n // 10)
    tr = tr_real[nvr:] + tr_f[nvr:]
    va = tr_real[:nvr] + tr_f[:nvr]
    return tr, va


def main():
    ap = argparse.ArgumentParser(description="CF generational generalization")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    d = load(args.seed, args.backbone)
    y, fam = d["label"], d["family"]
    print(f"dataset: {len(y)} imgs, real {int((y == 0).sum())} / fake {int((y == 1).sum())}")

    # real -> reference bank / train / test (disjoint)
    real = (y == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real)
    ref_idx, rest = real[:REF_N], real[REF_N:]
    tr_real, te_real = rest[:len(rest) // 2], rest[len(rest) // 2:]
    A = arms(d, ref_idx)

    # in-distribution: all generations mixed
    fake = (y == 1).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(fake)
    te_fake, pool_fake = fake[:len(fake) // 2], fake[len(fake) // 2:]
    tr, va = balanced_train(tr_real, pool_fake, rng)
    indist = evaluate(A, y, tr, va, te_real + te_fake)
    print("\n=== in-distribution (all generations) ===")
    print("  " + "  ".join(f"{k} {v:.4f}" for k, v in indist.items()))

    # leave-one-generation-out
    print("\n=== leave-one-generation-out (test AUC) ===")
    print(f"{'held-out':>12} | {'raw':>7} {'LID':>7} {'raw+LID':>8} | n_fake")
    means = {k: [] for k in A}
    logo = {}
    for g in GENERATIONS:
        g_fake = [i for i in range(len(y)) if y[i] == 1 and fam[i] == g]
        if not g_fake:
            continue
        other = [i for i in range(len(y)) if y[i] == 1 and fam[i] != g]
        tr, va = balanced_train(tr_real, other, rng)
        r = evaluate(A, y, tr, va, te_real + g_fake)
        for k in A:
            means[k].append(r[k])
        logo[g] = r
        print(f"{g:>12} | {r['raw']:>7.3f} {r['LID']:>7.3f} {r['raw+LID']:>8.3f} | {len(g_fake)}")
    print(f"{'MEAN':>12} | " + " ".join(f"{sum(v) / len(v):>7.3f}" for v in means.values()))

    save_json({"backbone": args.backbone, "seed": args.seed, "in_distribution": indist,
               "leave_one_generation_out": logo}, f"cf_generational_{args.backbone}")
    arm_names = ["raw", "LID", "raw+LID"]
    fams = list(logo.keys())
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for k, (arm, col) in enumerate(zip(arm_names, ["#5b8ff9", "#d1495b", "#61ddaa"])):
        ax.bar(x + (k - 1) * 0.27, [logo[g][arm] for g in fams], 0.27, label=arm, color=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels(fams, rotation=45, ha="right")
    ax.set_ylabel("held-out AUC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    ax.set_title(f"LOGO: LID adds nothing to a raw probe ({args.backbone})", fontsize=10)
    save_fig(fig, f"cf_generational_{args.backbone}")


if __name__ == "__main__":
    main()
