"""Selective LID, out-of-distribution, multi-seed. Reuses the LOO feature pool;
per held-out generator, trains raw and raw+LID heads on the other generators and
evaluates the confidence-gated SelectiveLID detector. Seeds vary the real split,
per-fold sampling, and head init. Reports mean +/- std over seeds for the AUC
gain on the raw-uncertain tail and the deployed detector vs. the raw baseline."""
import argparse
import copy
import random
import statistics as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

from utils.lid_estimator import compute_lid_features
from utils.selective import SelectiveLID

POOL = "./checkpoints/loo_cache/pool.pt"
GATE = 0.2                       # fraction of test routed to raw+LID
RAW_L, LID_L, K, REF_SIZE = 12, 7, 20, 1000


def fit(X, y, tr, va, seed, epochs=20):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(),
                      nn.Dropout(0.3), nn.Linear(128, 2))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best, bs, stale = -1.0, None, 0
    for _ in range(epochs):
        m.train()
        for b in torch.tensor(tr)[torch.randperm(len(tr))].split(64):
            opt.zero_grad()
            F.cross_entropy(m(X[b]), y[b]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            a = roc_auc_score(y[va], F.softmax(m(X[va]), 1)[:, 1])
        if a > best:
            best, bs, stale = a, copy.deepcopy(m.state_dict()), 0
        else:
            stale += 1
            if stale >= 4:
                break
    m.load_state_dict(bs)
    return m.eval()


def prob(m, X):
    with torch.no_grad():
        return F.softmax(m(X), 1)[:, 1]


def auc(yy, pp):
    try:
        return roc_auc_score(yy, pp)
    except ValueError:
        return float("nan")


def run_seed(seed, feats, gen, y, raw_all):
    rng = random.Random(seed)
    real_pos = (y == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real_pos)
    ref_pos, rest = real_pos[:REF_SIZE], real_pos[REF_SIZE:]
    h = len(rest) // 2
    tr_real, te_real = rest[:h], rest[h:]
    ref = feats[LID_L][ref_pos]
    both_all = torch.cat([raw_all, compute_lid_features(feats[LID_L], ref, k=K)], 1)

    acc = {k: [] for k in ("raw_auc", "sel_auc", "raw_acc", "sel_acc", "d10", "d20")}
    for g in sorted({v for v in gen.tolist() if v != 0}):
        g_pos = ((y == 1) & (gen == g)).nonzero(as_tuple=True)[0].tolist()
        heldin = ((y == 1) & (gen != g)).nonzero(as_tuple=True)[0].tolist()
        rng.shuffle(heldin)
        nvr = max(1, len(tr_real) // 10)
        va_real, tr_r = tr_real[:nvr], tr_real[nvr:]
        tr = tr_r + heldin[:len(tr_r)]
        va = va_real + heldin[len(tr_r):len(tr_r) + nvr]
        te = torch.tensor(te_real + g_pos)

        raw_m, both_m = fit(raw_all, y, tr, va, seed), fit(both_all, y, tr, va, seed)
        p_raw, p_both = prob(raw_m, raw_all[te]), prob(both_m, both_all[te])
        y_te = y[te]
        order = (p_raw - 0.5).abs().argsort()
        for frac, key in ((0.1, "d10"), (0.2, "d20")):
            idx = order[:int(len(te) * frac)]
            acc[key].append(auc(y_te[idx], p_both[idx]) - auc(y_te[idx], p_raw[idx]))

        tau = SelectiveLID.tau_for(prob(raw_m, raw_all[torch.tensor(va)]), GATE)
        p_sel = SelectiveLID(raw_m, both_m, tau).scores(raw_all[te], both_all[te])
        acc["raw_auc"].append(auc(y_te, p_raw))
        acc["sel_auc"].append(auc(y_te, p_sel))
        acc["raw_acc"].append(accuracy_score(y_te, (p_raw >= 0.5).int()))
        acc["sel_acc"].append(accuracy_score(y_te, (p_sel >= 0.5).int()))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description="Selective LID, multi-seed")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    blob = torch.load(POOL, map_location="cpu", weights_only=False)
    feats, gen, y = blob["feats"], blob["gen"], blob["y"]
    raw_all = feats[RAW_L]

    runs = [run_seed(s, feats, gen, y, raw_all) for s in range(args.seeds)]

    def ms(key):
        v = [r[key] for r in runs]
        return sum(v) / len(v), (st.stdev(v) if len(v) > 1 else 0.0)

    print(f"=== Selective LID, {args.seeds} seeds (mean +/- std over held-out generators) ===")
    print(f"raw LOO AUC        : {ms('raw_auc')[0]:.4f} +/- {ms('raw_auc')[1]:.4f}")
    print(f"AUC gain bottom 10%: {ms('d10')[0]:+.4f} +/- {ms('d10')[1]:.4f}")
    print(f"AUC gain bottom 20%: {ms('d20')[0]:+.4f} +/- {ms('d20')[1]:.4f}")
    print(f"deployed AUC       : raw {ms('raw_auc')[0]:.4f} -> sel {ms('sel_auc')[0]:.4f} "
          f"(+/-{ms('sel_auc')[1]:.4f})")
    print(f"deployed accuracy  : raw {ms('raw_acc')[0]:.4f} -> sel {ms('sel_acc')[0]:.4f} "
          f"(+/-{ms('sel_acc')[1]:.4f})")


if __name__ == "__main__":
    main()
