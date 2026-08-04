"""Per-family real->fake geometry: shift magnitude, pairwise cosine, separability."""
import argparse
import statistics as st

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils.cf_data import GENERATIONS, fake_index, load
from utils.figs import plt, save_fig, save_json


def per_seed(seed, backbone):
    d = load(seed, backbone)
    y, fam, X = d["label"], d["family"], d["feat12"]
    fk = fake_index(y, fam)
    fams = list(fk)
    mu = X[y == 0].mean(0)
    shift = {g: X[torch.tensor(v)].mean(0) - mu for g, v in fk.items()}
    udir = {g: shift[g] / shift[g].norm() for g in fams}
    std_real = {g: ((X[y == 0] - mu) @ udir[g]).std().item() for g in fams}

    mag = {g: shift[g].norm().item() for g in fams}
    off_cos = [torch.dot(udir[a], udir[b]).item()
               for a in fams for b in fams if a != b]
    sep = {(a, b): torch.dot(udir[a], shift[b]).item() / std_real[a]
           for a in fams for b in fams}
    # 1-D shared axis, in-sample by construction; whiten.py has the held-out version
    common = torch.stack([udir[g] for g in fams]).mean(0)
    common = common / common.norm()
    proj = (X - mu) @ common
    real_idx = (y == 0).nonzero(as_tuple=True)[0].tolist()
    shared_auc = {}
    for g, v in fk.items():
        m = torch.tensor(real_idx + v)
        shared_auc[g] = roc_auc_score(y[m], proj[m])
    return mag, sum(off_cos) / len(off_cos), sep, shared_auc, fams


def main():
    ap = argparse.ArgumentParser(description="Direction-geometry diagnostic")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--backbone", default="clip")
    args = ap.parse_args()
    R = [per_seed(s, args.backbone) for s in range(args.seeds)]
    fams = [g for g in GENERATIONS if all(g in r[4] for r in R)]
    ms = lambda xs: (sum(xs) / len(xs), st.stdev(xs) if len(xs) > 1 else 0.0)

    mag = {g: list(ms([r[0][g] for r in R])) for g in fams}
    off = list(ms([r[1] for r in R]))
    sep = {f"{a}|{b}": list(ms([r[2][(a, b)] for r in R])) for a in fams for b in fams}
    shared = {g: list(ms([r[3][g] for r in R])) for g in fams}

    print(f"direction geometry, mean +/- std over {args.seeds} seeds\n")
    print("shift magnitude ||mu_fake(F) - mu_real||:")
    for g in fams:
        print(f"  {g:>12}: {mag[g][0]:.3f} ± {mag[g][1]:.3f}")
    print(f"\nmean off-diagonal direction cosine: {off[0]:.3f} ± {off[1]:.3f}")
    print("\n1-D shared-axis detection AUC (in-sample, real vs family fake):")
    for g in fams:
        print(f"  {g:>12}: {shared[g][0]:.3f} ± {shared[g][1]:.3f}")

    save_json({"backbone": args.backbone, "families": fams, "seeds": args.seeds,
               "shift_magnitude": mag, "off_diagonal_cosine": off, "separability": sep,
               "shared_axis_auc": shared}, f"cf_directions_{args.backbone}")
    plot(mag, shared, sep, fams, args.backbone)


def plot(mag, shared, sep, fams, backbone):
    x = range(len(fams))

    # figure 1: two measures, two panels -- they share no unit, so they share no axis
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].bar(list(x), [mag[g][0] for g in fams], 0.6,
              yerr=[mag[g][1] for g in fams], color="#5b8ff9")
    ax[0].set_ylabel("||mu_fake - mu_real||")
    ax[0].set_title("Per-family shift magnitude", fontsize=10)
    ax[1].bar(list(x), [shared[g][0] for g in fams], 0.6,
              yerr=[shared[g][1] for g in fams], color="#d1495b")
    ax[1].set_ylabel("1-D shared-axis AUC (in-sample)")
    ax[1].set_ylim(0.5, 1.0)
    ax[1].set_title("Detectability on the shared axis", fontsize=10)
    for a in ax:
        a.set_xticks(list(x))
        a.set_xticklabels(fams, rotation=45, ha="right")
    save_fig(fig, f"cf_directions_{backbone}")

    # figure 2: asymmetric effective-separability matrix
    A = np.array([[sep[f"{a}|{b}"][0] for b in fams] for a in fams])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(A, cmap="magma")
    ax.set_xticks(list(x))
    ax.set_xticklabels(fams, rotation=45, ha="right")
    ax.set_yticks(list(x))
    ax.set_yticklabels(fams)
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
