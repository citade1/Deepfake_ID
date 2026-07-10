"""Multi-seed error bars for the fixed-feature experiments (Stage 1 ablation,
JPEG robustness, untrained-backbone ablation). Reuses cached features; only the
head init/training varies across seeds (features and splits are fixed)."""
import argparse
import copy
import os
import statistics as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from utils.lid_estimator import compute_lid_features

RAW_L, LID_L, K = 12, 7, 20


def load(cache, tag):
    b = torch.load(os.path.join(cache, f"{tag}.pt"), map_location="cpu", weights_only=False)
    return b["feats"], b["labels"]


def arms(feats, ref):
    raw = feats[RAW_L]
    lid = compute_lid_features(feats[LID_L], ref, k=K)
    return {"raw": raw, "LID": lid, "raw+LID": torch.cat([raw, lid], 1)}


def fit(Xtr, ytr, Xva, yva, seed, epochs=20):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                      nn.Dropout(0.3), nn.Linear(128, 2))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best, bs, stale = -1.0, None, 0
    for _ in range(epochs):
        m.train()
        for i in torch.randperm(len(ytr)).split(64):
            opt.zero_grad()
            F.cross_entropy(m(Xtr[i]), ytr[i]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            a = roc_auc_score(yva, F.softmax(m(Xva), 1)[:, 1])
        if a > best:
            best, bs, stale = a, copy.deepcopy(m.state_dict()), 0
        else:
            stale += 1
            if stale >= 4:
                break
    m.load_state_dict(bs)
    return m.eval()


def test_auc(m, X, y):
    with torch.no_grad():
        return roc_auc_score(y, F.softmax(m(X), 1)[:, 1])


def ms(vals):
    return sum(vals) / len(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def ablation(cache, seeds, title):
    ftr, ytr = load(cache, "train")
    fva, yva = load(cache, "val")
    fte, yte = load(cache, "test")
    ref = load(cache, "ref")[0][LID_L]
    Atr, Ava, Ate = arms(ftr, ref), arms(fva, ref), arms(fte, ref)
    print(f"\n=== {title} ({seeds} seeds) ===")
    for name in Atr:
        m, s = ms([test_auc(fit(Atr[name], ytr, Ava[name], yva, sd), Ate[name], yte)
                   for sd in range(seeds)])
        print(f"  {name:>8}: {m:.4f} +/- {s:.4f}")


def jpeg(cache, seeds):
    ftr, ytr = load(cache, "train")
    fva, yva = load(cache, "val")
    ref = load(cache, "ref")[0][LID_L]
    heads = [fit(ftr[RAW_L], ytr, fva[RAW_L], yva, sd) for sd in range(seeds)]
    qualities = [("clean", os.path.join(cache, "test.pt"))]
    qualities += [(f"q{q}", os.path.join(cache, "jpeg", f"q{q}.pt")) for q in (95, 75, 50, 30, 15)]
    print(f"\n=== JPEG robustness, raw AUC ({seeds} seeds) ===")
    for tag, path in qualities:
        b = torch.load(path, map_location="cpu", weights_only=False)
        X, y = b["feats"][RAW_L], b["labels"]
        m, s = ms([test_auc(h, X, y) for h in heads])
        print(f"  {tag:>6}: {m:.4f} +/- {s:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Multi-seed error bars")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    ablation("./checkpoints/stage1_cache", args.seeds, "Stage 1 ablation (in-distribution)")
    jpeg("./checkpoints/stage1_cache", args.seeds)
    ablation("./checkpoints/untrained_cache", args.seeds, "Untrained (random-init) backbone")


if __name__ == "__main__":
    main()
