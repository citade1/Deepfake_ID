"""In-distribution difficulty and leave-one-generation-out generalization on CF."""
import argparse
import random
import statistics as st

import numpy as np
import torch

from utils.cf_data import (FAKE_FIT, FAKE_TEST, GENERATIONS, LID_BANK, balanced, fake_index,
                           load, real_split, train_val)
from utils.figs import plt, save_fig, save_json
from utils.heads import auc, fit_head
from utils.lid_estimator import compute_lid_features

K = 20
ARMS = ["raw", "LID", "raw+LID"]


def arms(d, ref_idx):
    raw = d["feat12"]
    lid = compute_lid_features(d["feat7"], d["feat7"][ref_idx], k=K)
    return {"raw": raw, "LID": lid, "raw+LID": torch.cat([raw, lid], dim=1)}


def evaluate(A, y, tr, va, te):
    tr, va, te = torch.tensor(tr), torch.tensor(va), torch.tensor(te)
    return {name: auc(fit_head(X[tr], y[tr], X[va], y[va], seed=0), X[te], y[te])
            for name, X in A.items()}


def per_seed(seed, backbone):
    rng = random.Random(seed)
    d = load(seed, backbone)
    y, fam = d["label"], d["family"]
    real = (y == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real)
    bank, rest = real[:LID_BANK], real[LID_BANK:]         # the bank must not reach the test side
    real_tr, real_te = real_split(rest)
    A = arms(d, bank)
    fk = fake_index(y, fam, rng)

    fit_all = [i for v in fk.values() for i in v[:FAKE_FIT]]
    te_all = [i for v in fk.values() for i in v[FAKE_FIT:FAKE_FIT + FAKE_TEST]]
    tr, va = train_val(balanced(real_tr, fit_all, rng), rng)
    indist = evaluate(A, y, tr, va, real_te + te_all)

    logo = {}
    for G in fk:
        seen = [i for g, v in fk.items() if g != G for i in v[:FAKE_FIT]]
        tr, va = train_val(balanced(real_tr, seen, rng), rng)
        logo[G] = evaluate(A, y, tr, va, real_te + fk[G][FAKE_FIT:FAKE_FIT + FAKE_TEST])
    return indist, logo


def main():
    ap = argparse.ArgumentParser(description="CF generational generalization")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    IND, LOGO = zip(*[per_seed(s, args.backbone) for s in range(args.seeds)])
    fams = [g for g in GENERATIONS if all(g in r for r in LOGO)]
    ms = lambda v: [sum(v) / len(v), st.stdev(v) if len(v) > 1 else 0.0]

    indist = {a: ms([r[a] for r in IND]) for a in ARMS}
    logo = {g: {a: ms([r[g][a] for r in LOGO]) for a in ARMS} for g in fams}

    print(f"mean +/- std over {args.seeds} seeds ({args.backbone})")
    print(f"\n{'test set':>12} | " + " ".join(f"{a:>13}" for a in ARMS))
    print(f"{'in-dist':>12} | " + " ".join(f"{indist[a][0]:.3f}±{indist[a][1]:.3f}" for a in ARMS))
    for g in fams:
        print(f"{g:>12} | " + " ".join(f"{logo[g][a][0]:.3f}±{logo[g][a][1]:.3f}" for a in ARMS))

    save_json({"backbone": args.backbone, "seeds": args.seeds, "in_distribution": indist,
               "leave_one_generation_out": logo}, f"cf_generational_{args.backbone}")
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for k, (arm, col) in enumerate(zip(ARMS, ["#5b8ff9", "#d1495b", "#61ddaa"])):
        ax.bar(x + (k - 1) * 0.27, [logo[g][arm][0] for g in fams], 0.27,
               yerr=[logo[g][arm][1] for g in fams], label=arm, color=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels(fams, rotation=45, ha="right")
    ax.set_ylabel("held-out AUC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    ax.set_title(f"LOGO: does LID add to a raw probe? ({args.backbone})", fontsize=10)
    save_fig(fig, f"cf_generational_{args.backbone}")


if __name__ == "__main__":
    main()
