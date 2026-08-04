"""LayerNorm fine-tune (leave-one-generation-out): does an unseen generator's
representation join the fake cluster while decision transfer degrades? See README."""
import argparse
import collections
import io
import os
import random
import shutil
import statistics as st
import tempfile

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPVisionModel

from experiments.prepare_cf import REPO, family, select_shards
from utils.figs import plt, save_fig, save_json
from utils.geometry import shrink_cov, unit
from utils.heads import auc as head_auc
from utils.heads import fit_head, prob_fake

BACKBONE = "openai/clip-vit-base-patch16"
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
GEN = ["GAN", "PixelDiff", "LatentDiff", "Flow", "Commercial"]
MODERN = ["LatentDiff", "Flow", "Commercial"]


def ok_image(b):
    try:
        return min(Image.open(io.BytesIO(b)).size) >= 8   # skip degenerate 1x1 images
    except Exception:
        return False


def gather(shards, per_family, real_per_source, cache):
    """Download shards -> (bytes, family, label). Real is capped PER SOURCE so
    FFHQ faces are present (keeps GAN hard); fakes capped per family. Disk-cached."""
    if os.path.exists(cache):
        pool = torch.load(cache, weights_only=False)
        print(f"loaded cached pool ({len(pool)} imgs) from {cache}")
        return pool
    pool, fake_n, real_n = [], collections.Counter(), collections.Counter()
    for subset, fname in shards:
        tmp = tempfile.mkdtemp()
        try:
            path = hf_hub_download(REPO[subset], fname, repo_type="dataset", local_dir=tmp)
            d = pq.read_table(path, columns=["image_data", "model_name"]).to_pydict()
            for b, mn in zip(d["image_data"], d["model_name"]):
                fam = family(mn)
                if not ok_image(b):
                    continue
                if fam == "REAL":
                    if real_n[mn] >= real_per_source.get(mn, 0):
                        continue
                    real_n[mn] += 1
                    pool.append((b, "REAL", 0))
                else:
                    if fake_n[fam] >= per_family:
                        continue
                    fake_n[fam] += 1
                    pool.append((b, fam, 1))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print("real by source:", dict(real_n))
    torch.save(pool, cache)
    return pool


class DS(Dataset):
    def __init__(self, pool, proc):
        self.pool, self.proc = pool, proc

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, i):
        b, _, lab = self.pool[i]
        im = Image.open(io.BytesIO(b)).convert("RGB")
        return self.proc(images=im, return_tensors="pt")["pixel_values"][0], lab


def cls12(model, px):
    return F.normalize(model(pixel_values=px, output_hidden_states=True)
                       .hidden_states[12][:, 0, :], dim=1)


@torch.no_grad()
def feats(model, proc, pool, batch=32):
    model.eval()
    out = []
    for i in range(0, len(pool), batch):
        ims = [Image.open(io.BytesIO(b)).convert("RGB") for b, _, _ in pool[i:i + batch]]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(DEVICE)
        out.append(cls12(model, px).cpu())
    return torch.cat(out)


def masks(pool):
    y = torch.tensor([lab for _, _, lab in pool])
    fam = [f for _, f, _ in pool]
    return y, fam


def real_anchored(X, pool, held, seen):
    """Held family's cos to the seen common axis + its 1-D detection AUC."""
    y, fam = masks(pool)
    mu = X[y == 0].mean(0)
    udir = {}
    for g in GEN:
        m = torch.tensor([fam[i] == g and y[i] == 1 for i in range(len(y))])
        if m.sum() >= 5:
            udir[g] = unit(X[m].mean(0) - mu)
    axis = unit(torch.stack([udir[g] for g in seen if g in udir]).mean(0))
    cos = torch.dot(udir[held], axis).item()
    hm = torch.tensor([fam[i] == held and y[i] == 1 for i in range(len(y))])
    detect = torch.tensor([(fam[i] == held and y[i] == 1) or y[i] == 0 for i in range(len(y))])
    proj = (X - mu) @ axis
    return cos, roc_auc_score(y[detect], proj[detect]), (X[hm].mean(0) - mu).norm().item()


def fake_membership(X, pool, held, seen):
    """Anchor-free: is the held-out family nearer the SEEN-FAKE cluster or the REAL
    cluster? Returns mean cos to each, and delta (>0 = on the fake side)."""
    y, fam = masks(pool)
    real_dir = unit(X[y == 0].mean(0))
    sf = torch.tensor([fam[i] in seen and y[i] == 1 for i in range(len(y))])
    fake_dir = unit(X[sf].mean(0))
    hm = torch.tensor([fam[i] == held and y[i] == 1 for i in range(len(y))])
    Xh = X[hm]
    tf, tr = (Xh @ fake_dir).mean().item(), (Xh @ real_dir).mean().item()
    return tf, tr, tf - tr


def disc_geometry(Xe, Xt, eval_pool, train_pool, held, seen):
    """Q1: alignment of a held-out family's shift with the seen raw axis d vs cleaned axis w."""
    ytr, ftr = masks(train_pool)
    rtr = Xt[torch.tensor([ytr[i] == 0 for i in range(len(ytr))])]
    ftr_s = Xt[torch.tensor([ftr[i] in seen and ytr[i] == 1 for i in range(len(ytr))])]
    d_seen = ftr_s.mean(0) - rtr.mean(0)
    S = 0.5 * (shrink_cov(rtr) + shrink_cov(ftr_s))
    w = unit(torch.linalg.solve(S, d_seen))             # Sigma^-1 d_seen
    ds = unit(d_seen)
    ye, fe = masks(eval_pool)
    mre = Xe[torch.tensor([ye[i] == 0 for i in range(len(ye))])].mean(0)
    mhe = Xe[torch.tensor([fe[i] == held and ye[i] == 1 for i in range(len(ye))])].mean(0)
    d_held = unit(mhe - mre)
    return torch.dot(d_held, ds).item(), torch.dot(d_held, w).item()


def maha_auc(Xe, Xt, eval_pool, train_pool, held):
    """Q2: detect by Mahalanobis distance to real only (no fake boundary); higher = more fake."""
    ytr = masks(train_pool)[0]
    rtr = Xt[torch.tensor([ytr[i] == 0 for i in range(len(ytr))])]
    mur = rtr.mean(0)
    Sinv = torch.linalg.inv(shrink_cov(rtr, alpha=0.5))   # heavier shrinkage: tuned real can collapse
    ye, fe = masks(eval_pool)
    detect = torch.tensor([(fe[i] == held and ye[i] == 1) or ye[i] == 0 for i in range(len(ye))])
    diff = Xe[detect] - mur
    score = torch.einsum("ni,ij,nj->n", diff, Sinv, diff)
    return roc_auc_score(ye[detect], score)


def fit_seen_head(Xtr, pool_tr, seen):
    """Linear head trained on SEEN fakes vs real (held-out family absent from train)."""
    ytr, ftr = masks(pool_tr)
    keep = torch.tensor([(ftr[i] in seen and ytr[i] == 1) or ytr[i] == 0 for i in range(len(ytr))])
    idx = keep.nonzero(as_tuple=True)[0]
    idx = idx[torch.randperm(len(idx))]
    nv = len(idx) // 10
    return fit_head(Xtr[idx[nv:]], ytr[idx[nv:]], Xtr[idx[:nv]], ytr[idx[:nv]], seed=0)


def finetune(model, proc, pool, epochs, bs, lr, patience=3, val_frac=0.12):
    """LayerNorm-only fine-tune with a held-out val split + early stopping, so the
    stopping point (hence the tuned features) is consistent across seeds."""
    for m in model.modules():
        for p in m.parameters(recurse=False):
            p.requires_grad_(isinstance(m, nn.LayerNorm))
    head = nn.Linear(768, 2).to(DEVICE)
    tp = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tp + list(head.parameters()), lr=lr)

    idx = list(range(len(pool)))
    random.Random(0).shuffle(idx)                          # fixed val split (not a variance source)
    nv = max(bs, int(len(pool) * val_frac))
    dl = DataLoader(DS([pool[i] for i in idx[nv:]], proc), batch_size=bs, shuffle=True, num_workers=0)
    va = DataLoader(DS([pool[i] for i in idx[:nv]], proc), batch_size=bs, num_workers=0)

    def snap():
        return ([p.detach().clone() for p in tp], {k: v.clone() for k, v in head.state_dict().items()})

    best, best_state, stale, stop = float("inf"), snap(), 0, 0
    for ep in range(epochs):
        model.train()
        for px, lab in dl:
            px, lab = px.to(DEVICE), lab.to(DEVICE)
            loss = F.cross_entropy(head(cls12(model, px)), lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(F.cross_entropy(head(cls12(model, px.to(DEVICE))), lab.to(DEVICE)).item() * len(lab)
                     for px, lab in va) / nv
        stop = ep + 1
        if vl < best - 1e-4:
            best, best_state, stale = vl, snap(), 0
        else:
            stale += 1
            if stale >= patience:
                break
    for p, q in zip(tp, best_state[0]):                     # restore best-val weights
        p.data.copy_(q)
    head.load_state_dict(best_state[1])
    print(f"  stopped ep {stop}, best val loss {best:.4f}", flush=True)


def metrics(Xe, Xt, eval_pool, train_pool, held, seen):
    """All readouts for one held-out family in feature space (Xe eval, Xt train).
    Returns aggregate dict + per-held-image (head prob_fake, fake-side delta)."""
    cos, auc1d, mag = real_anchored(Xe, eval_pool, held, seen)
    tf, tr, delta = fake_membership(Xe, eval_pool, held, seen)
    mlp = fit_seen_head(Xt, train_pool, seen)
    y, fam = masks(eval_pool)
    detect = torch.tensor([(fam[i] == held and y[i] == 1) or y[i] == 0 for i in range(len(y))])
    tauc = head_auc(mlp, Xe[detect], y[detect])

    hm = torch.tensor([fam[i] == held and y[i] == 1 for i in range(len(y))])
    Xh = Xe[hm]
    probs = prob_fake(mlp, Xh)
    real_dir = unit(Xe[y == 0].mean(0))
    sf = torch.tensor([fam[i] in seen and y[i] == 1 for i in range(len(y))])
    fake_dir = unit(Xe[sf].mean(0))
    pdelta = (Xh @ fake_dir) - (Xh @ real_dir)          # per-image fake-side score
    cos_gen, cos_disc = disc_geometry(Xe, Xt, eval_pool, train_pool, held, seen)
    maha = maha_auc(Xe, Xt, eval_pool, train_pool, held)
    agg = dict(cos=cos, auc1d=auc1d, mag=mag, delta=delta, tauc=tauc,
               cos_gen=cos_gen, cos_disc=cos_disc, maha=maha)
    return agg, probs, pdelta


def run_holdout(model, init_state, proc, eval_pool, train_all, Xe0, Xt0, held, epochs, bs, lr, seed=0):
    """Frozen vs LN-tuned readouts for one held-out family; model reset to init_state first."""
    seen = [g for g in GEN if g != held]
    train_pool = [x for x in train_all if x[1] != held]
    gf, pf, df = metrics(Xe0, Xt0, eval_pool, train_all, held, seen)

    torch.manual_seed(seed)                             # same init per family (fair, reproducible)
    model.load_state_dict(init_state)                   # reset to pretrained weights
    model.to(DEVICE)
    finetune(model, proc, train_pool, epochs, bs, lr)
    Xe1, Xt1 = feats(model, proc, eval_pool), feats(model, proc, train_all)
    gt, pt, dt = metrics(Xe1, Xt1, eval_pool, train_all, held, seen)

    hard = (pf > 0.3) & (pf < 0.7)                       # frozen-ambiguous held images
    easy = ~hard
    def mv(t, m):
        return t[m].mean().item() if m.any() else float("nan")
    hardrow = dict(n_hard=int(hard.sum()),
                   d_hard=(mv(df, hard), mv(dt, hard)), p_hard=(mv(pf, hard), mv(pt, hard)),
                   d_easy=(mv(df, easy), mv(dt, easy)), p_easy=(mv(pf, easy), mv(pt, easy)))
    return gf, gt, hardrow


MK = ["delta", "tauc", "maha", "cos_gen", "cos_disc", "cos", "auc1d", "mag"]


def main():
    ap = argparse.ArgumentParser(description="LN fine-tune LOGO: do unseen families join fakes or drift to real?")
    ap.add_argument("--holdout", default="all", help="family to hold out, or 'all'")
    ap.add_argument("--seeds", type=int, default=3, help="splits + fine-tune seeds for error bars")
    ap.add_argument("--per-family", type=int, default=700)
    ap.add_argument("--real", type=int, default=2400)
    ap.add_argument("--faces-frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=15)   # early stopping cuts this short
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-frac", type=float, default=0.35)
    args = ap.parse_args()

    nf = int(args.real * args.faces_frac)
    real_src = {"FFHQ": nf, "COCO": (args.real - nf) // 2, "LandscapesHQ": (args.real - nf) // 2}
    shards = select_shards({g: args.per_family for g in GEN},
                           [("FFHQ", 1), ("COCO", 1), ("LandscapesHQ", 1)])
    print(f"{len(shards)} shards | device {DEVICE} | holdout={args.holdout} | seeds={args.seeds}")

    proc = CLIPImageProcessor.from_pretrained(BACKBONE, local_files_only=True)
    cache = f"checkpoints/cf_cache/ft_pool_bal_pf{args.per_family}_r{args.real}_f{nf}.pt"
    pool = gather(shards, args.per_family, real_src, cache)
    print("gathered:", dict(collections.Counter(f for _, f, _ in pool)))

    model = CLIPVisionModel.from_pretrained(BACKBONE, local_files_only=True)
    init_state = {k: v.clone() for k, v in model.state_dict().items()}   # pretrained snapshot
    model.to(DEVICE)
    Xpool = feats(model, proc, pool)                       # frozen feats for whole pool, once
    n_eval = int(len(pool) * args.eval_frac)
    holdouts = GEN if args.holdout == "all" else [args.holdout]

    acc = {h: {k: {"frozen": [], "tuned": []} for k in MK} for h in holdouts}
    for s in range(args.seeds):
        idx = list(range(len(pool)))
        random.Random(s).shuffle(idx)
        ev, tr = idx[:n_eval], idx[n_eval:]
        eval_pool, Xe0 = [pool[i] for i in ev], Xpool[torch.tensor(ev)]
        train_all, Xt0 = [pool[i] for i in tr], Xpool[torch.tensor(tr)]
        for held in holdouts:
            gf, gt, _ = run_holdout(model, init_state, proc, eval_pool, train_all, Xe0, Xt0,
                                    held, args.epochs, args.bs, args.lr, seed=s)
            for k in MK:
                acc[held][k]["frozen"].append(gf[k])
                acc[held][k]["tuned"].append(gt[k])
            print(f"[seed {s}] {held:>12}: delta {gf['delta']:.3f}->{gt['delta']:.3f}  "
                  f"headAUC {gf['tauc']:.3f}->{gt['tauc']:.3f}  maha {gf['maha']:.3f}->{gt['maha']:.3f}",
                  flush=True)

    def ms(xs):
        return [sum(xs) / len(xs), st.stdev(xs) if len(xs) > 1 else 0.0]
    data = {h: {k: {"frozen": ms(acc[h][k]["frozen"]), "tuned": ms(acc[h][k]["tuned"])} for k in MK}
            for h in holdouts}

    print(f"\n=== SUMMARY: frozen -> tuned, mean +/- std over {args.seeds} seeds ===")
    print(f"{'held-out':>12} | {'delta':>17} | {'head AUC':>17} | {'Maha AUC':>17}")
    for h in holdouts:
        c = lambda k: (f"{data[h][k]['frozen'][0]:.2f}±{data[h][k]['frozen'][1]:.2f}->"
                       f"{data[h][k]['tuned'][0]:.2f}±{data[h][k]['tuned'][1]:.2f}")
        print(f"{h:>12} | {c('delta'):>17} | {c('tauc'):>17} | {c('maha'):>17}")

    save_json({"seeds": args.seeds, "families": holdouts, "metrics": data}, "cf_finetune")
    plot_finetune(data, holdouts)


def plot_finetune(data, fams):
    yy = list(range(len(fams)))

    def bell(ax, key, title, color="#d1495b"):
        gm = [data[h][key]["frozen"][0] for h in fams]
        gs = [data[h][key]["frozen"][1] for h in fams]
        tm = [data[h][key]["tuned"][0] for h in fams]
        ts = [data[h][key]["tuned"][1] for h in fams]
        for i in yy:
            ax.plot([gm[i], tm[i]], [i, i], "-", color="#ccc", zorder=1)
        ax.errorbar(gm, yy, xerr=gs, fmt="o", color="#888", label="frozen", zorder=2, capsize=2)
        ax.errorbar(tm, yy, xerr=ts, fmt="o", color=color, label="LN-tuned", zorder=2, capsize=2)
        ax.set_yticks(yy)
        ax.set_yticklabels(fams)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="lower left")

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    bell(ax[0], "delta", "fake-side membership (>0 = fake)")
    ax[0].axvline(0, color="k", lw=0.8)
    # Q1: generative (tuned) vs discriminative (tuned) axis alignment, with std
    gg = [data[h]["cos_gen"]["tuned"] for h in fams]
    dd = [data[h]["cos_disc"]["tuned"] for h in fams]
    ax[1].errorbar([v[0] for v in gg], [i + 0.12 for i in yy], xerr=[v[1] for v in gg], fmt="o",
                   color="#5b8ff9", capsize=2, label="cos to d_seen (generative)")
    ax[1].errorbar([v[0] for v in dd], [i - 0.12 for i in yy], xerr=[v[1] for v in dd], fmt="o",
                   color="#e8a33d", capsize=2, label="cos to w (discriminative)")
    ax[1].set_yticks(yy)
    ax[1].set_yticklabels(fams)
    ax[1].set_title("Q1: axis alignment (tuned)", fontsize=10)
    ax[1].legend(fontsize=7, loc="lower center")
    bell(ax[2], "tauc", "seen-trained head AUC")
    bell(ax[3], "maha", "Q2: Mahalanobis-to-real AUC")
    fig.suptitle("LN fine-tune (LOGO), mean±std: representation aligns (generative) but "
                 "decision transfer does not improve", fontsize=11)
    save_fig(fig, "cf_finetune")


if __name__ == "__main__":
    main()
