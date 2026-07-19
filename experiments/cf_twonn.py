"""Global manifold ID (TwoNN) of real vs each fake family at CLIP layers 7/12 --
a between-class ID gap, distinct from the refuted per-sample LID. See README."""
import argparse
import random
import statistics as st

import numpy as np
import torch

from utils.cf_data import GENERATIONS, load
from utils.figs import plt, save_fig, save_json
from utils.lid_estimator import twonn_global_id

LAYERS = ["feat7", "feat12"]


def per_seed(seed, n, backbone):
    d = load(seed, backbone)
    y, fam = d["label"], d["family"]
    groups = {"REAL": [i for i in range(len(y)) if y[i] == 0]}
    for g in GENERATIONS:
        groups[g] = [i for i in range(len(y)) if y[i] == 1 and fam[i] == g]
    out = {}
    for layer in LAYERS:
        X = d[layer]
        row = {}
        for name, idx in groups.items():
            idx = list(idx)
            random.Random(seed).shuffle(idx)
            idx = idx[:n]
            if len(idx) >= 200:
                row[name] = twonn_global_id(X[torch.tensor(idx)])
        out[layer] = row
    return out


def main():
    ap = argparse.ArgumentParser(description="Global manifold ID (TwoNN) per family")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=1000, help="samples per group (equalized)")
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    R = [per_seed(s, args.n, args.backbone) for s in range(args.seeds)]
    names = ["REAL"] + [g for g in GENERATIONS if g in R[0]["feat12"]]

    data = {}
    for layer in LAYERS:
        data[layer] = {nm: [sum(r[layer][nm] for r in R) / len(R),
                            st.stdev([r[layer][nm] for r in R]) if len(R) > 1 else 0.0]
                       for nm in names if nm in R[0][layer]}

    print(f"global TwoNN ID, mean +/- std over {args.seeds} seeds, N={args.n} per group")
    for layer in LAYERS:
        print(f"\n{layer}:")
        for nm in names:
            if nm in data[layer]:
                print(f"  {nm:>12}: {data[layer][nm][0]:5.2f} ± {data[layer][nm][1]:.2f}")

    save_json({"backbone": args.backbone, "n_per_group": args.n, "seeds": args.seeds,
               "twonn_id": data}, f"cf_twonn_{args.backbone}")
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for k, (layer, col) in enumerate(zip(LAYERS, ["#5b8ff9", "#d1495b"])):
        ax.bar(x + (k - 0.5) * 0.4, [data[layer].get(nm, [np.nan])[0] for nm in names], 0.4,
               yerr=[data[layer].get(nm, [0, 0])[1] for nm in names], label=layer, color=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("global TwoNN intrinsic dim")
    ax.legend(fontsize=8)
    ax.set_title(f"Manifold intrinsic dim: real vs each family ({args.backbone})", fontsize=10)
    save_fig(fig, f"cf_twonn_{args.backbone}")


if __name__ == "__main__":
    main()
