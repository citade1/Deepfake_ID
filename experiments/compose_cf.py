"""Draw a labelled, generator-balanced dataset from the CF cache. See README."""
import argparse
import collections
import glob
import random

import torch

from utils.cf_data import family

FACE_SOURCE = "FFHQ"
DEFAULT_FAKE = {"GAN": 1500, "PixelDiff": 2500, "LatentDiff": 6000, "Flow": 3000, "Commercial": 3000}


def load_pool(backbone):
    """Concatenate all cached shards into flat tensors + per-image generator tags.
    Same seed across backbones draws identical images (shards share row order)."""
    f7, f12, gens = [], [], []
    for path in sorted(glob.glob(f"./checkpoints/cf_cache/{backbone}/shards/*.pt")):
        b = torch.load(path, map_location="cpu", weights_only=False)
        f7.append(b["f7"]); f12.append(b["f12"]); gens += b["model_name"]
    return torch.cat(f7), torch.cat(f12), gens


def balanced_draw(indices_by_key, total, rng):
    """Take `total` indices spread as evenly as possible across the keys."""
    keys = [k for k, v in indices_by_key.items() if v]
    for v in indices_by_key.values():
        rng.shuffle(v)
    picked, cursor = [], collections.Counter()
    while len(picked) < total and keys:
        for k in list(keys):
            if cursor[k] < len(indices_by_key[k]):
                picked.append(indices_by_key[k][cursor[k]]); cursor[k] += 1
                if len(picked) >= total:
                    break
            else:
                keys.remove(k)
    return picked


def main():
    ap = argparse.ArgumentParser(description="Compose a CF dataset draw")
    ap.add_argument("--backbone", default="clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", type=int, default=12000)
    ap.add_argument("--face-frac", type=float, default=0.5)
    ap.add_argument("--fake", default="", help="e.g. LatentDiff=6000,Flow=3000")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    fake_targets = DEFAULT_FAKE if not args.fake else \
        {k: int(v) for k, v in (kv.split("=") for kv in args.fake.split(","))}

    f7, f12, gens = load_pool(args.backbone)
    fam = [family(g) for g in gens]

    # index images by (family -> generator -> [row ids])
    real_by_source, fake_by_gen = collections.defaultdict(list), collections.defaultdict(lambda: collections.defaultdict(list))
    for i, (fm, g) in enumerate(zip(fam, gens)):
        if fm == "REAL":
            real_by_source[g].append(i)
        else:
            fake_by_gen[fm][g].append(i)

    rows, label, fam_tag, gen_tag = [], [], [], []

    # fakes: per family, balanced across its generators
    for fm, tgt in fake_targets.items():
        idx = balanced_draw(fake_by_gen[fm], tgt, rng)
        rows += idx; label += [1] * len(idx)
        fam_tag += [fm] * len(idx); gen_tag += [gens[i] for i in idx]

    # real: faces (FFHQ) vs the rest, at the requested fraction
    n_face = int(args.real * args.face_frac)
    faces = real_by_source.get(FACE_SOURCE, [])[:]
    rng.shuffle(faces)
    face_idx = faces[:n_face]
    rest_sources = {s: v for s, v in real_by_source.items() if s != FACE_SOURCE}
    rest_idx = balanced_draw(rest_sources, args.real - len(face_idx), rng)
    for i in face_idx + rest_idx:
        rows.append(i); label.append(0)
        fam_tag.append("REAL"); gen_tag.append(gens[i])

    rows = torch.tensor(rows)
    out = {
        "feat7": f7[rows], "feat12": f12[rows],
        "label": torch.tensor(label), "family": fam_tag, "generator": gen_tag,
        "seed": args.seed,
    }
    path = f"checkpoints/cf_cache/dataset_{args.backbone}_s{args.seed}.pt"
    torch.save(out, path)

    print(f"composed {len(label)} imgs ({args.backbone}, seed {args.seed}) -> {path}")
    print("  label:", dict(collections.Counter(label)))
    print("  family:", dict(collections.Counter(fam_tag)))
    print("  real sources:", dict(collections.Counter(g for g, l in zip(gen_tag, label) if l == 0)))
    for fm in fake_targets:
        gcount = collections.Counter(g for g, t in zip(gen_tag, fam_tag) if t == fm)
        print(f"  {fm}: {len(gcount)} generators, e.g. {dict(list(gcount.most_common(5)))}")


if __name__ == "__main__":
    main()
