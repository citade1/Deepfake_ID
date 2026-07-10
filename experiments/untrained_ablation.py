"""Stage 2, failure mode 3 — trained vs. untrained backbone (Ansuini et al.).
Same probes on a RANDOM-init CLIP (frozen): if random features still separate
real/fake, the signal is raw image statistics, not training-induced semantics."""
import argparse
import os

import torch
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from experiments.stage1_ablation import DEVICE, extract, train_head
from utils.dataloader_helper import make_splits
from utils.lid_estimator import compute_lid_features

CACHE = "./checkpoints/untrained_cache"


def cached(model, proc, ds, layers, batch, tag, meta):
    path = os.path.join(CACHE, f"{tag}.pt")
    if os.path.exists(path):
        b = torch.load(path, map_location="cpu", weights_only=False)
        if b["meta"] == meta:
            print(f"  {tag}: cached")
            return b["feats"], b["labels"]
    feats, y = extract(model, proc, ds, layers, batch, f"extract {tag}")
    torch.save({"feats": feats, "labels": y, "meta": meta}, path)
    return feats, y


def main():
    ap = argparse.ArgumentParser(description="Untrained-backbone ablation")
    ap.add_argument("--backbone", default="openai/clip-vit-base-patch32")
    ap.add_argument("--lid-layer", type=int, default=7)
    ap.add_argument("--raw-layer", type=int, default=12)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--ref-size", type=int, default=1024)
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--max-eval", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    proc = CLIPImageProcessor.from_pretrained(args.backbone)
    torch.manual_seed(0)
    model = CLIPVisionModel(CLIPVisionConfig.from_pretrained(args.backbone)).to(DEVICE).eval()
    print(f"RANDOM-init {args.backbone} | lid@{args.lid_layer} raw@{args.raw_layer}")

    tr, va, te, ref_ds = make_splits(ref_size=args.ref_size, seed=42,
                                     max_train=args.max_train, max_eval=args.max_eval)
    layers = [args.lid_layer, args.raw_layer]
    meta = dict(rand=True, layers=layers, max_train=args.max_train, max_eval=args.max_eval)

    ref = cached(model, proc, ref_ds, [args.lid_layer], args.batch,
                 "ref", {**meta, "s": "ref"})[0][args.lid_layer]

    def arms(f):
        return f[args.raw_layer], compute_lid_features(f[args.lid_layer], ref, k=args.k)

    ftr, ytr = cached(model, proc, tr, layers, args.batch, "train", {**meta, "s": "train"})
    fva, yva = cached(model, proc, va, layers, args.batch, "val", {**meta, "s": "val"})
    fte, yte = cached(model, proc, te, layers, args.batch, "test", {**meta, "s": "test"})
    raw_tr, lid_tr = arms(ftr)
    raw_va, lid_va = arms(fva)
    raw_te, lid_te = arms(fte)

    print("\n=== Untrained (random-init) backbone ===")
    train_head("raw", raw_tr, ytr, raw_va, yva, raw_te, yte)
    train_head("LID", lid_tr, ytr, lid_va, yva, lid_te, yte)
    train_head("raw+LID", torch.cat([raw_tr, lid_tr], 1), ytr,
               torch.cat([raw_va, lid_va], 1), yva, torch.cat([raw_te, lid_te], 1), yte)
    print("(trained CLIP for comparison: raw 0.974 / LID 0.626 / raw+LID 0.975)")


if __name__ == "__main__":
    main()
