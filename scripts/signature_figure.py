"""Build the one figure that carries the study, from the analysis JSONs. No re-run needed."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cf_data import GENERATIONS
from utils.figs import plt, save_fig

BACKBONE = sys.argv[1] if len(sys.argv) > 1 else "clip"
# validated for CVD: worst adjacent-pair separation dE 9.0 (protan/deutan/tritan)
COLOR = dict(zip(GENERATIONS, ["#5b8ff9", "#d1495b", "#61ddaa", "#e8a33d", "#8c6bb1", "#4d4d4d"]))
INK, MUTED = "#222222", "#777777"

wh = json.load(open(f"figures/cf_whiten_{BACKBONE}.json"))
sub = json.load(open(f"figures/cf_subspace_{BACKBONE}.json"))

PANELS = [
    ("What the method buys", ["generative_d", "whitened_w", "full_mlp"], wh["auc"],
     ["raw axis\n$d$", "whitened\n$w=\\Sigma^{-1}d$", f"full MLP\n{wh['dim']}-d"]),
    ("What the dimension buys", ["shared_1d", "subspace", "full"], sub["auc"],
     ["1-D\nshared axis", f"{sub['dims']['subspace']}-D\nsubspace", f"full\n{sub['dims']['full']}-d"]),
]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
for ax, (title, keys, data, ticks) in zip(axes, PANELS):
    x = range(len(keys))
    for fam in GENERATIONS:
        if fam not in data:
            continue
        y = [data[fam][k][0] for k in keys]
        e = [data[fam][k][1] for k in keys]
        lead = fam == "GAN"
        ax.errorbar(x, y, yerr=e, color=COLOR[fam], lw=2.6 if lead else 1.6,
                    marker="o", ms=9 if lead else 7, capsize=2, zorder=3 if lead else 2,
                    alpha=1.0 if lead else 0.8, label=fam)
        if lead and ax is axes[0]:
            ax.annotate("GAN", (x[0], y[0]), xytext=(10, -2), textcoords="offset points",
                        va="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ticks, fontsize=9, color=INK)
    ax.set_xlim(-0.35, len(keys) - 0.45)
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("held-out AUC", color=INK)
axes[0].set_ylim(0.55, 1.0)
axes[1].legend(fontsize=8, ncol=2, loc="lower right", frameon=False)
fig.suptitle("Five of six generator families are captured by one closed-form, low-dimensional "
             "axis — GAN is not", fontsize=12.5, color=INK)
fig.supxlabel(f"Community Forensics, leave-one-generation-out, {BACKBONE} ViT-B/16 frozen, "
              f"5 seeds (bars = std)", fontsize=8.5, color=MUTED)
save_fig(fig, f"signature_{BACKBONE}")
print(f"figures/signature_{BACKBONE}.png")
