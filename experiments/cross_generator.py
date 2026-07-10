"""Stage 2 — leave-one-generator-out cross-generalization. CLIP features are
extracted once for a balanced pool, then for each held-out generator we train
raw / LID / raw+LID heads on the other generators (+real, balanced) and test on
the held-out one. Reports per-generator and mean AUC — how well the
in-distribution signal transfers to unseen generators."""
import argparse
import copy
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

from experiments.stage1_ablation import DEVICE
from utils.dataloader_helper import DATASET_STR, Collate
from utils.lid_estimator import compute_lid_features

CACHE = "./checkpoints/loo_cache"


@torch.no_grad()
def extract_pool(model, proc, ds, layers, batch):
    loader = DataLoader(ds, batch_size=batch, collate_fn=Collate(proc))
    store = {l: [] for l in layers}
    for inp, _ in tqdm(loader, desc="extract pool"):
        hs = model(pixel_values=inp["pixel_values"].to(DEVICE),
                   output_hidden_states=True).hidden_states
        for l in layers:
            store[l].append(F.normalize(hs[l][:, 0, :], dim=1).cpu())
    return {l: torch.cat(v) for l, v in store.items()}


def fit_eval(X, y, tr, va, te, epochs=20, seed=42):
    torch.manual_seed(seed)
    mlp = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(128, 2))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    def auc(idx):
        mlp.eval()
        with torch.no_grad():
            p = F.softmax(mlp(X[idx]), dim=1)[:, 1]
        return roc_auc_score(y[idx], p)

    best, best_state, stale = -1.0, None, 0
    for _ in range(epochs):
        mlp.train()
        for b in torch.tensor(tr)[torch.randperm(len(tr))].split(64):
            opt.zero_grad()
            F.cross_entropy(mlp(X[b]), y[b]).backward()
            opt.step()
        a = auc(va)
        if a > best:
            best, best_state, stale = a, copy.deepcopy(mlp.state_dict()), 0
        else:
            stale += 1
            if stale >= 4:
                break
    mlp.load_state_dict(best_state)
    return auc(te)


def main():
    ap = argparse.ArgumentParser(description="Leave-one-generator-out")
    ap.add_argument("--backbone", default="openai/clip-vit-base-patch32")
    ap.add_argument("--lid-layer", type=int, default=7)
    ap.add_argument("--raw-layer", type=int, default=12)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--cap-real", type=int, default=6000)
    ap.add_argument("--cap-fake", type=int, default=2000)
    ap.add_argument("--ref-size", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    proc = CLIPImageProcessor.from_pretrained(args.backbone)
    model = CLIPVisionModel.from_pretrained(args.backbone).to(DEVICE).eval()

    ds = load_dataset(DATASET_STR, split="train")
    names = ds.features["generator"].names
    real_id = names.index("Real")
    all_gen = ds["generator"]
    fake_gens = sorted({g for g in all_gen if g != real_id})

    rng = random.Random(42)
    by = defaultdict(list)
    for i, g in enumerate(all_gen):
        by[g].append(i)
    pool = []
    for g, lst in by.items():
        rng.shuffle(lst)
        pool += lst[:(args.cap_real if g == real_id else args.cap_fake)]
    rng.shuffle(pool)
    sub = ds.select(pool)
    gen, y = torch.tensor(sub["generator"]), torch.tensor(sub["label"])
    layers = [args.lid_layer, args.raw_layer]

    meta = dict(backbone=args.backbone, layers=layers, n=len(sub),
                cap_real=args.cap_real, cap_fake=args.cap_fake)
    path = os.path.join(CACHE, "pool.pt")
    feats = None
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob["meta"] == meta and torch.equal(blob["gen"], gen):
            feats = blob["feats"]
            print("pool: cached")
    if feats is None:
        feats = extract_pool(model, proc, sub, layers, args.batch)
        torch.save({"feats": feats, "gen": gen, "y": y, "meta": meta}, path)

    real_pos = (y == 0).nonzero(as_tuple=True)[0].tolist()
    rng.shuffle(real_pos)
    ref_pos, rest = real_pos[:args.ref_size], real_pos[args.ref_size:]
    h = len(rest) // 2
    tr_real, te_real = rest[:h], rest[h:]
    ref = feats[args.lid_layer][ref_pos]

    raw_all = feats[args.raw_layer]
    lid_all = compute_lid_features(feats[args.lid_layer], ref, k=args.k)
    feat_sets = {"raw": raw_all, "LID": lid_all,
                 "raw+LID": torch.cat([raw_all, lid_all], dim=1)}

    print(f"\n=== Leave-one-generator-out AUC ===\n"
          f"{'held-out':>11} {'raw':>7} {'LID':>7} {'raw+LID':>8}")
    scores = {k: [] for k in feat_sets}
    for g in fake_gens:
        g_pos = ((y == 1) & (gen == g)).nonzero(as_tuple=True)[0].tolist()
        heldin = ((y == 1) & (gen != g)).nonzero(as_tuple=True)[0].tolist()
        rng.shuffle(heldin)
        nvr = max(1, len(tr_real) // 10)
        va_real, tr_r = tr_real[:nvr], tr_real[nvr:]
        tr_f = heldin[:len(tr_r)]                       # balance train 1:1
        va_f = heldin[len(tr_r):len(tr_r) + nvr]
        tr, va, te = tr_r + tr_f, va_real + va_f, te_real + g_pos
        line = [fit_eval(X, y, tr, va, te) for X in feat_sets.values()]
        for name, s in zip(feat_sets, line):
            scores[name].append(s)
        print(f"{names[g]:>11} {line[0]:>7.4f} {line[1]:>7.4f} {line[2]:>8.4f}")
    print(f"{'MEAN':>11} " + " ".join(f"{sum(v)/len(v):>7.4f}" for v in scores.values()))


if __name__ == "__main__":
    main()
