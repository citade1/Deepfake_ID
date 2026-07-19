"""Per-family real->fake direction geometry: shift magnitudes, pairwise cosines,
and the effective separability that reproduces the transfer asymmetry. See README."""
import argparse
import statistics as st

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils.cf_data import GENERATIONS, load
from utils.figs import plt, save_fig, save_json


def per_seed(seed, backbone):
    d = load(seed, backbone)
    y, fam, X = d["label"], d["family"], d["feat12"]
    mu = X[y == 0].mean(0)
    shift = {}
    for g in GENERATIONS:
        m = torch.tensor([fam[i] == g and y[i] == 1 for i in range(len(y))])
        shift[g] = X[m].mean(0) - mu
    udir = {g: shift[g] / shift[g].norm() for g in GENERATIONS}
    std_real = {g: ((X[y == 0] - mu) @ udir[g]).std().item() for g in GENERATIONS}

    mag = {g: shift[g].norm().item() for g in GENERATIONS}
    off_cos = [torch.dot(udir[a], udir[b]).item()
               for a in GENERATIONS for b in GENERATIONS if a != b]
    sep = {(a, b): torch.dot(udir[a], shift[b]).item() / std_real[a]
           for a in GENERATIONS for b in GENERATIONS}
    # 1-D shared axis (mean direction) detection AUC per family
    common = torch.stack([udir[g] for g in GENERATIONS]).mean(0)
    common = common / common.norm()
    proj = (X - mu) @ common
    shared_auc = {}
    for g in GENERATIONS:
        m = torch.tensor([(fam[i] == g and y[i] == 1) or y[i] == 0 for i in range(len(y))])
        shared_auc[g] = roc_auc_score(y[m], proj[m])
    return mag, sum(off_cos) / len(off_cos), sep, shared_auc


def main():
    ap = argparse.ArgumentParser(description="Direction-geometry diagnostic")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    R = [per_seed(s, args.backbone) for s in range(args.seeds)]
    ms = lambda xs: (sum(xs) / len(xs), st.stdev(xs) if len(xs) > 1 else 0.0)

    mag = {g: list(ms([r[0][g] for r in R])) for g in GENERATIONS}
    off = list(ms([r[1] for r in R]))
    sep = {f"{a}|{b}": list(ms([r[2][(a, b)] for r in R]))
           for a in GENERATIONS for b in GENERATIONS}
    shared = {g: list(ms([r[3][g] for r in R])) for g in GENERATIONS}

    print(f"direction geometry, mean +/- std over {args.seeds} seeds\n")
    print("shift magnitude ||mu_fake(F) - mu_real||:")
    for g in GENERATIONS:
        print(f"  {g:>12}: {mag[g][0]:.3f} ± {mag[g][1]:.3f}")
    print(f"\nmean off-diagonal direction cosine: {off[0]:.3f} ± {off[1]:.3f}")
    print("\n1-D shared-axis detection AUC (real vs family fake):")
    for g in GENERATIONS:
        print(f"  {g:>12}: {shared[g][0]:.3f} ± {shared[g][1]:.3f}")

    save_json({"backbone": args.backbone, "families": GENERATIONS, "seeds": args.seeds,
               "shift_magnitude": mag, "off_diagonal_cosine": off, "separability": sep,
               "shared_axis_auc": shared}, f"cf_directions_{args.backbone}")
    plot(mag, shared, sep, args.backbone)


def plot(mag, shared, sep, backbone):
    x = range(len(GENERATIONS))

    # figure 1: per-family shift magnitude + 1-D shared-axis detectability
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar([i - 0.2 for i in x], [mag[g][0] for g in GENERATIONS], 0.4,
           yerr=[mag[g][1] for g in GENERATIONS], label="shift magnitude", color="#5b8ff9")
    ax2 = ax.twinx()
    ax2.bar([i + 0.2 for i in x], [shared[g][0] for g in GENERATIONS], 0.4,
            yerr=[shared[g][1] for g in GENERATIONS], label="1-D axis AUC", color="#d1495b")
    ax2.grid(False)
    ax.set_xticks(list(x))
    ax.set_xticklabels(GENERATIONS, rotation=45, ha="right")
    ax.set_ylabel("shift magnitude ||mu_fake - mu_real||")
    ax2.set_ylabel("1-D shared-axis AUC")
    ax2.set_ylim(0.5, 1.0)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Per-family shift & 1-D detectability", fontsize=10)
    save_fig(fig, f"cf_directions_{backbone}")

    # figure 2: asymmetric effective-separability matrix
    A = np.array([[sep[f"{a}|{b}"][0] for b in GENERATIONS] for a in GENERATIONS])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(A, cmap="magma")
    ax.set_xticks(list(x))
    ax.set_xticklabels(GENERATIONS, rotation=45, ha="right")
    ax.set_yticks(list(x))
    ax.set_yticklabels(GENERATIONS)
    ax.set_xlabel("test family B")
    ax.set_ylabel("train axis A")
    for i in x:
        for j in x:
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="effective separability")
    ax.set_title("Effective separability (asymmetric)", fontsize=10)
    save_fig(fig, f"cf_separability_{backbone}")


if __name__ == "__main__":
    main()
