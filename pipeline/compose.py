"""Draw a labelled, generator-balanced dataset from the CF cache. See README."""
import argparse
import collections
import glob
import os
import random

import torch

from utils import paths as P
from utils.cf_data import FACE_SOURCE, GENERATIONS, family, norm_source, real_dataset

# equal per family so no single architecture dominates the draw
DEFAULT_FAKE = {f: 1500 for f in GENERATIONS}


def load_pool(backbone):
    """All cached shards as flat tensors + metadata, cached so repeated draws stay cheap."""
    shards = sorted(glob.glob(f"{P.shards_dir(backbone)}/*.pt"))
    if not shards:
        raise FileNotFoundError(f"no features for {backbone!r}; run pipeline/extract.py first")
    # same file names survive a re-extract, so the preproc tag is part of the key
    want = torch.load(shards[0], map_location="cpu", weights_only=False, mmap=True).get("preproc")
    cache = P.pool_file(backbone)
    if os.path.exists(cache):
        p = torch.load(cache, map_location="cpu", weights_only=False)
        if p.get("shards") == shards and p.get("preproc") == want:
            return p["f7"], p["f12"], p["gens"], p["fams"], p["srcs"]
    f7, f12, gens, fams, srcs, quals = [], [], [], [], [], set()
    for path in shards:
        b = torch.load(path, map_location="cpu", weights_only=False)
        quals.add(b.get("preproc"))
        f7.append(b["f7"]); f12.append(b["f12"]); gens += b["model_name"]
        srcs += [real_dataset(m, rs) if l == 0 else norm_source(rs)
                 for m, rs, l in zip(b["model_name"], b["real_source"], b["label"])]
        fams += [family(m, a, l) for m, a, l in zip(b["model_name"], b["architecture"], b["label"])]
    if len(quals) > 1:
        raise ValueError(f"shards mix preprocessing {quals}; re-extract {backbone!r}")
    f7, f12 = torch.cat(f7), torch.cat(f12)
    torch.save({"f7": f7, "f12": f12, "gens": gens, "fams": fams,
                "srcs": srcs, "shards": shards, "preproc": want}, P.ensure(cache))
    return f7, f12, gens, fams, srcs


def balanced_draw(indices_by_key, total, rng):
    """Take `total` indices spread as evenly as possible across the keys."""
    pools = {k: list(v) for k, v in indices_by_key.items() if v}   # copy: caller keeps its order
    for v in pools.values():
        rng.shuffle(v)
    picked = []
    for i in range(max(map(len, pools.values()), default=0)):      # round i takes each pool's i-th
        if len(picked) >= total:
            break
        for v in pools.values():
            if i < len(v):                                         # a pool drops out once spent
                picked.append(v[i])
                if len(picked) >= total:
                    break
    return picked


def main():
    ap = argparse.ArgumentParser(description="Compose a CF dataset draw")
    ap.add_argument("--backbone", default="clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", type=int, default=7000)
    ap.add_argument("--face-frac", type=float, default=0.5)
    ap.add_argument("--fake", default="", help="e.g. LatDiff=6000,Flow=3000")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    fake_targets = DEFAULT_FAKE if not args.fake else \
        {k: int(v) for k, v in (kv.split("=") for kv in args.fake.split(","))}

    # shard order matches across backbones, so one --seed draws the same images for all
    f7, f12, gens, fam, srcs = load_pool(args.backbone)

    # index images by (family -> generator -> [row ids])
    real_by_source, fake_by_gen = collections.defaultdict(list), collections.defaultdict(lambda: collections.defaultdict(list))
    for i, (fm, g) in enumerate(zip(fam, gens)):
        if fm == "REAL":
            real_by_source[srcs[i]].append(i)
        else:
            fake_by_gen[fm][g].append(i)

    if real_by_source.get("unknown"):
        print(f"WARNING: {len(real_by_source['unknown'])} real images have an unresolved source")

    rows, label, fam_tag, gen_tag = [], [], [], []

    # fakes: per family, balanced across its generators
    short = {}
    for fm, tgt in fake_targets.items():
        idx = balanced_draw(fake_by_gen[fm], tgt, rng)
        if len(idx) < tgt:
            short[fm] = (len(idx), tgt)
        rows += idx; label += [1] * len(idx)
        fam_tag += [fm] * len(idx); gen_tag += [gens[i] for i in idx]

    # real: faces (FFHQ) vs the rest, at the requested fraction
    n_face = int(args.real * args.face_frac)
    faces = real_by_source.get(FACE_SOURCE, [])[:]
    rng.shuffle(faces)
    face_idx = faces[:n_face]
    if len(face_idx) < n_face:
        short[f"{FACE_SOURCE} faces"] = (len(face_idx), n_face)   # backfilled, so the total hides it
    rest_sources = {s: v for s, v in real_by_source.items() if s != FACE_SOURCE}
    rest_idx = balanced_draw(rest_sources, args.real - len(face_idx), rng)
    if len(face_idx) + len(rest_idx) < args.real:
        short["REAL"] = (len(face_idx) + len(rest_idx), args.real)
    for i in face_idx + rest_idx:
        rows.append(i); label.append(0)
        fam_tag.append("REAL"); gen_tag.append(gens[i])

    rows = torch.tensor(rows)
    out = {
        "feat7": f7[rows], "feat12": f12[rows],
        "label": torch.tensor(label), "family": fam_tag, "generator": gen_tag,
        "real_source": [srcs[i] for i in rows.tolist()],   # for content-controlled subsets
        # grouped family by family then REAL: never take a positional slice of this
        "seed": args.seed,
    }
    path = P.ensure(P.dataset_file(args.backbone, args.seed))
    torch.save(out, path)

    if short:
        print("WARNING: target not met -- extract more shards for: "
              + ", ".join(f"{k} {a}/{b}" for k, (a, b) in short.items()))
    print(f"composed {len(label)} imgs ({args.backbone}, seed {args.seed}) -> {path}")
    print("  label:", dict(collections.Counter(label)))
    print("  family:", dict(collections.Counter(fam_tag)))
    print("  real sources:", dict(collections.Counter(
        s for s, l in zip(out["real_source"], label) if l == 0)))
    for fm in fake_targets:
        gcount = collections.Counter(g for g, t in zip(gen_tag, fam_tag) if t == fm)
        print(f"  {fm}: {len(gcount)} generators, e.g. {dict(list(gcount.most_common(5)))}")


if __name__ == "__main__":
    main()
