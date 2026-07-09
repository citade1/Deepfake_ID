"""TwoNN ID per CLIP layer (Ansuini "hunchback") on a balanced real/fake sample,
in cosine geometry. Reports the ID-peak layer (choice (a)) and the layer of
largest real/fake ID gap. Picks the LID layer for Stage 1."""
import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

from utils.dataloader_helper import DATASET_STR
from utils.lid_estimator import twonn_global_id

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


def load_balanced(per_class):
    """Stream until per_class real and per_class fake are collected (bounded
    download — only reads shards until the sample is full)."""
    ds = load_dataset(DATASET_STR, split="train", streaming=True)
    real, fake = [], []
    for r in ds:
        bucket = real if r["label"] == 0 else fake
        if len(bucket) < per_class:
            bucket.append(r["image"])
        if len(real) >= per_class and len(fake) >= per_class:
            break
    print(f"streamed -> real {len(real)} / fake {len(fake)}")
    n = min(per_class, len(real), len(fake))
    images = real[:n] + fake[:n]
    labels = torch.tensor([0] * n + [1] * n)
    return images, labels


@torch.no_grad()
def extract_all_layers(model, processor, images, batch):
    store = None
    for i in tqdm(range(0, len(images), batch), desc="CLIP layers"):
        imgs = [im.convert("RGB") if im.mode != "RGB" else im
                for im in images[i:i + batch]]
        inp = processor(images=imgs, return_tensors="pt").to(DEVICE)
        hs = model(**inp, output_hidden_states=True).hidden_states  # (L+1) x (B,T,D)
        if store is None:
            store = [[] for _ in hs]
        for l, h in enumerate(hs):
            store[l].append(h[:, 0, :].cpu())                       # CLS token
    return [torch.cat(s) for s in store]


def main():
    ap = argparse.ArgumentParser(description="CLIP ID profile across layers")
    ap.add_argument("--backbone", default="openai/clip-vit-base-patch32")
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    print(f"Device: {DEVICE} | backbone: {args.backbone}")

    processor = CLIPImageProcessor.from_pretrained(args.backbone)
    model = CLIPVisionModel.from_pretrained(args.backbone).to(DEVICE).eval()

    images, labels = load_balanced(args.per_class)
    real_m, fake_m = labels == 0, labels == 1
    print(f"sample: {len(images)} (real {real_m.sum().item()} / fake {fake_m.sum().item()})")

    per_layer = extract_all_layers(model, processor, images, args.batch)

    print(f"\n{'layer':>5} | {'ID_all':>7} {'ID_real':>7} {'ID_fake':>7} {'|gap|':>6}")
    profile = []
    for l, feat in enumerate(per_layer):
        fn = F.normalize(feat, dim=1)                     # cosine geometry
        id_all = twonn_global_id(fn)
        id_real = twonn_global_id(fn[real_m])
        id_fake = twonn_global_id(fn[fake_m])
        gap = abs(id_real - id_fake)
        profile.append((l, id_all, id_real, id_fake, gap))
        print(f"{l:>5} | {id_all:7.2f} {id_real:7.2f} {id_fake:7.2f} {gap:6.2f}")

    valid = [r for r in profile if not (r[1] != r[1])]  # drop NaN (layer 0 = constant CLS)
    peak = max(valid, key=lambda r: r[1])
    maxgap = max(valid, key=lambda r: r[4])
    print(f"\n(a) ID-peak layer      = {peak[0]}  (ID_all {peak[1]:.2f})")
    print(f"    max real/fake gap  = layer {maxgap[0]}  (|gap| {maxgap[4]:.2f})")
    print("    -> use ID-peak as the LID layer; report the gap layer alongside.")


if __name__ == "__main__":
    main()
