"""Community Forensics: its repos, and how `label`/`architecture` map to generator families."""
import torch

from utils import paths

# the two halves of the dataset; `-Eval` holds the paired real/fake evaluation split
REPO = {"small": "OwensLab/CommunityForensics-Small",
        "eval": "OwensLab/CommunityForensics-Eval"}

# CF's own `architecture` values, oldest -> newest, except Flow: CF files its flow-matching
# models under Commercial, so those three are split out by name.
GENERATIONS = ["GAN", "PixDiff", "LatDiff", "Flow", "Commercial", "Other"]
_FLOW = {"FLUX-dev", "FLUX-schnell", "LFM"}

# real dataset names as they appear in `model_name` (Small only); elsewhere the name comes
# from `real_source`. Sources are lower-cased everywhere because CF's casing is inconsistent.
REAL_IN_MODEL_NAME = {"ffhq", "coco", "landscapeshq", "vision"}
FACE_SOURCE = "ffhq"


def norm_source(s):
    return (s or "").strip().lower()


def real_dataset(model_name, real_source):
    """Which real dataset a label=0 row came from: Small names it in `model_name`, Eval in `real_source`."""
    m = norm_source(model_name)
    if m in REAL_IN_MODEL_NAME:
        return m
    rs = norm_source(real_source)
    return rs if rs and rs != "n/a" else "unknown"


def family(model_name, architecture, label):
    """Family of one image. `label` decides real/fake: an Eval real still carries a generator name."""
    if label == 0:
        return "REAL"
    if model_name in _FLOW:
        return "Flow"
    if architecture not in GENERATIONS:
        raise ValueError(f"unknown architecture {architecture!r} for {model_name!r}")
    return architecture


# one split budget for every analysis, so AUCs are comparable across files
REAL_FIT, REAL_TEST = 2500, 2500
FAKE_FIT, FAKE_TEST = 700, 700
LID_BANK = 1000


def fake_index(label, family, rng=None):
    """{family: [row ids]} over fakes, in GENERATIONS order, absent families dropped."""
    idx = {g: [] for g in GENERATIONS}
    for i, l in enumerate(label.tolist()):
        if l == 1:
            idx[family[i]].append(i)
    out = {g: v for g, v in idx.items() if v}
    for v in out.values():
        if rng:
            rng.shuffle(v)
    return out


def halves(idx):
    """A family's fakes as a fit half and a disjoint test half, or None if either is empty."""
    fit, test = idx[:FAKE_FIT], idx[FAKE_FIT:FAKE_FIT + FAKE_TEST]
    return (fit, test) if fit and test else None


def real_split(real, n_train=REAL_FIT, n_test=REAL_TEST):
    """Disjoint real fit/test halves. Raises rather than silently emptying the test side."""
    if len(real) < n_train + n_test:
        raise ValueError(f"this split needs {n_train + n_test} real images, draw has {len(real)}"
                         f" -- compose with --real {n_train + n_test} or more")
    return real[:n_train], real[n_train:n_train + n_test]


def balanced(reals, fakes, rng):
    """Equal-size draw from two index pools, so no probe is trained on a skewed prior."""
    a, b = list(reals), list(fakes)
    rng.shuffle(a); rng.shuffle(b)
    n = min(len(a), len(b))
    return a[:n] + b[:n]


def train_val(idx, rng, frac=0.1):
    """Shuffle and split off a validation fraction. -> (train, val)"""
    idx = list(idx)
    rng.shuffle(idx)
    nv = max(1, int(len(idx) * frac))
    return idx[nv:], idx[:nv]


def load(seed=0, backbone="clip"):
    d = torch.load(paths.dataset_file(backbone, seed),
                   map_location="cpu", weights_only=False)
    d["family"] = list(d["family"])
    d["generator"] = list(d["generator"])
    if "real_source" in d:
        d["real_source"] = list(d["real_source"])
    return d
