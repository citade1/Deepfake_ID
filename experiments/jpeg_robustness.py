"""JPEG robustness (Pope): does the CLIP raw-CLS detector survive compression?
Raw and LID heads are trained on clean features (reused from stage1 cache), then
evaluated on the same test images re-encoded as JPEG across quality levels.
Collapse under compression => the signal was low-level texture."""
import argparse
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import CLIPImageProcessor, CLIPVisionModel

from experiments.stage1_ablation import CACHE, DEVICE, extract, train_head
from utils.dataloader_helper import make_splits
from utils.lid_estimator import compute_lid_features


def load_cached(tag):
    b = torch.load(os.path.join(CACHE, f"{tag}.pt"), map_location="cpu", weights_only=False)
    return b["feats"], b["labels"]


@torch.no_grad()
def score(mlp, X, y):
    p = F.softmax(mlp(X), dim=1)[:, 1]
    return accuracy_score(y, (p >= 0.5).int()), roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser(description="JPEG robustness sweep")
    ap.add_argument("--backbone", default="openai/clip-vit-base-patch32")
    ap.add_argument("--lid-layer", type=int, default=7)
    ap.add_argument("--raw-layer", type=int, default=12)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--qualities", type=int, nargs="+", default=[95, 75, 50, 30, 15])
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    jpeg_dir = os.path.join(CACHE, "jpeg")
    os.makedirs(jpeg_dir, exist_ok=True)
    print(f"Device {DEVICE} | {args.backbone} | lid@{args.lid_layer} raw@{args.raw_layer}")

    processor = CLIPImageProcessor.from_pretrained(args.backbone)
    model = CLIPVisionModel.from_pretrained(args.backbone).to(DEVICE).eval()

    ref = load_cached("ref")[0][args.lid_layer]

    def arms(feats):
        raw = feats[args.raw_layer]
        lid = compute_lid_features(feats[args.lid_layer], ref, k=args.k)
        return raw, lid

    ftr, ytr = load_cached("train")
    fva, yva = load_cached("val")
    fte, yte = load_cached("test")
    raw_tr, lid_tr = arms(ftr)
    raw_va, lid_va = arms(fva)
    raw_te, lid_te = arms(fte)

    raw_mlp, _, _ = train_head("raw", raw_tr, ytr, raw_va, yva, raw_te, yte)
    lid_mlp, _, _ = train_head("LID", lid_tr, ytr, lid_va, yva, lid_te, yte)

    _, _, test_ds, _ = make_splits(ref_size=1024, seed=42, max_train=12000, max_eval=2000)
    layers = [args.lid_layer, args.raw_layer]

    rows = [("clean", *score(raw_mlp, raw_te, yte), *score(lid_mlp, lid_te, yte))]
    for q in args.qualities:
        path = os.path.join(jpeg_dir, f"q{q}.pt")
        if os.path.exists(path):
            b = torch.load(path, map_location="cpu", weights_only=False)
            feats, y = b["feats"], b["labels"]
        else:
            feats, y = extract(model, processor, test_ds, layers, args.batch,
                               f"jpeg q{q}", jpeg_quality=q)
            torch.save({"feats": feats, "labels": y}, path)
        raw_q, lid_q = arms(feats)
        rows.append((f"q{q}", *score(raw_mlp, raw_q, y), *score(lid_mlp, lid_q, y)))

    print("\n=== JPEG robustness ===")
    print(f"{'quality':>7} | {'raw acc':>7} {'raw auc':>7} | {'lid acc':>7} {'lid auc':>7}")
    for tag, ra, ru, la, lu in rows:
        print(f"{tag:>7} | {ra:7.4f} {ru:7.4f} | {la:7.4f} {lu:7.4f}")


if __name__ == "__main__":
    main()
