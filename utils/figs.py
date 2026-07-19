"""Save experiment outputs in paper-ready form: JSON data + 300-dpi PNG figures."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = "figures"
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight",
                     # auto-space panels/colorbars/suptitle so labels never crowd a neighbour
                     "figure.constrained_layout.use": True,
                     "figure.constrained_layout.w_pad": 0.18,
                     "figure.constrained_layout.h_pad": 0.12,
                     "figure.constrained_layout.wspace": 0.06})


def save_json(obj, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def save_fig(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path
