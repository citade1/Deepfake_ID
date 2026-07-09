"""Stage 1 — does LID add signal over the raw representation? Three arms on a
frozen CLIP (cosine geometry, in-distribution): raw = CLS@last, LID-only =
LID@peak-layer vs a real-image reference bank, raw+LID = concat. raw+LID vs raw
is the marginal contribution; LID-only says whether any signal exists at all."""
import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

from utils.dataloader_helper import Collate, make_splits
from utils.lid_estimator import compute_lid_features

CACHE = "./checkpoints/stage1_cache"
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def extract(model, processor, ds, layers, batch, desc, jpeg_quality=None):
    loader = DataLoader(ds, batch_size=batch,
                        collate_fn=Collate(processor, jpeg_quality=jpeg_quality))
    store, labels = {l: [] for l in layers}, []
    for inp, y in tqdm(loader, desc=desc):
        hs = model(pixel_values=inp["pixel_values"].to(DEVICE),
                   output_hidden_states=True).hidden_states
        for l in layers:
            store[l].append(F.normalize(hs[l][:, 0, :], dim=1).cpu())  # cosine geometry
        labels.append(y)
    return {l: torch.cat(v) for l, v in store.items()}, torch.cat(labels)


def cached_extract(model, processor, ds, layers, batch, tag, meta):
    path = os.path.join(CACHE, f"{tag}.pt")
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob["meta"] == meta:
            print(f"  {tag}: cached")
            return blob["feats"], blob["labels"]
    feats, labels = extract(model, processor, ds, layers, batch, f"extract {tag}")
    torch.save({"feats": feats, "labels": labels, "meta": meta}, path)
    return feats, labels


def train_head(name, Xtr, ytr, Xva, yva, Xte, yte, epochs=20):
    torch.manual_seed(42)
    mlp = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(128, 2))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    def evalset(X, y):
        mlp.eval()
        with torch.no_grad():
            p = F.softmax(mlp(X), dim=1)[:, 1]
        return accuracy_score(y, (p >= 0.5).int()), roc_auc_score(y, p)

    best, best_state, stale = -1.0, None, 0
    for _ in range(epochs):
        mlp.train()
        for i in torch.randperm(len(ytr)).split(64):
            opt.zero_grad()
            F.cross_entropy(mlp(Xtr[i]), ytr[i]).backward()
            opt.step()
        _, auc = evalset(Xva, yva)
        if auc > best:
            best, best_state, stale = auc, copy.deepcopy(mlp.state_dict()), 0
        else:
            stale += 1
            if stale >= 4:
                break
    mlp.load_state_dict(best_state)
    acc, auc = evalset(Xte, yte)
    print(f"  [{name:>9}] test acc {acc:.4f} auc {auc:.4f} (dim {Xtr.shape[1]}, best val auc {best:.4f})")
    return mlp, acc, auc


def main():
    ap = argparse.ArgumentParser(description="Stage 1 ablation: does LID add signal?")
    ap.add_argument("--backbone", default="openai/clip-vit-base-patch32")
    ap.add_argument("--lid-layer", type=int, default=7)
    ap.add_argument("--raw-layer", type=int, default=12)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--ref-size", type=int, default=1024)
    ap.add_argument("--max-train", type=int, default=12000)
    ap.add_argument("--max-eval", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    print(f"Device {DEVICE} | backbone {args.backbone} | lid@{args.lid_layer} raw@{args.raw_layer}")

    processor = CLIPImageProcessor.from_pretrained(args.backbone)
    model = CLIPVisionModel.from_pretrained(args.backbone).to(DEVICE).eval()

    train_ds, val_ds, test_ds, ref_ds = make_splits(
        ref_size=args.ref_size, seed=42, max_train=args.max_train, max_eval=args.max_eval)
    layers = sorted({args.lid_layer, args.raw_layer})
    meta = dict(backbone=args.backbone, layers=layers, ref_size=args.ref_size,
                max_train=args.max_train, max_eval=args.max_eval)

    ref_feats, _ = cached_extract(model, processor, ref_ds, [args.lid_layer], args.batch,
                                  "ref", {**meta, "split": "ref"})
    ref = ref_feats[args.lid_layer]
    splits = {}
    for tag, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        feats, y = cached_extract(model, processor, ds, layers, args.batch, tag,
                                  {**meta, "split": tag})
        raw = feats[args.raw_layer]
        lid = compute_lid_features(feats[args.lid_layer], ref, k=args.k)
        splits[tag] = dict(raw=raw, lid=lid, both=torch.cat([raw, lid], dim=1), y=y)

    print("\n=== Stage 1 ablation (in-distribution) ===")
    for name, key in [("raw", "raw"), ("LID-only", "lid"), ("raw+LID", "both")]:
        train_head(name,
                   splits["train"][key], splits["train"]["y"],
                   splits["val"][key],   splits["val"]["y"],
                   splits["test"][key],  splits["test"]["y"])


if __name__ == "__main__":
    main()
