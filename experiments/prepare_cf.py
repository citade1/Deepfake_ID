"""Download CF shards once and extract per-shard features for every study backbone
(resumable). See README."""
import argparse
import collections
import glob
import io
import json
import os
import shutil
import tempfile

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from utils import backbones as B

REPO = {"small": "OwensLab/CommunityForensics-Small", "eval": "OwensLab/CommunityForensics-Eval"}
CACHE = "./checkpoints/cf_cache"
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

REAL_SOURCES = {"COCO", "LandscapesHQ", "FFHQ", "VISION"}


def shard_dir(backbone):
    return f"{CACHE}/{backbone}/shards"


def family(gen):
    g = gen.lower()
    if gen in REAL_SOURCES:
        return "REAL"
    if any(k in g for k in ["midjourney", "dalle", "imagen", "firefly", "ideogram"]):
        return "Commercial"
    if "flux" in g or g == "lfm":
        return "Flow"
    if any(k in gen for k in ["GAN", "StyleS", "StyleGAN", "ProGAN", "BigGAN", "CIPS",
                              "Gansformer", "GALIP", "Hourglass", "StyleSwin", "ProjectedGAN"]):
        return "GAN"
    if any(k in g for k in ["glide", "guideddiffusion", "vqdiffusion", "deepfloyd", "dit", "taming"]):
        return "PixelDiff"
    return "LatentDiff"


def basename(path_or_cache):
    # parquet basename, shared key between download name and cache name
    return os.path.basename(path_or_cache).split("__")[-1].replace(".pt", "")


def select_shards(targets, real_shards):
    """Greedily pick shards covering each fake family to its target image count,
    plus the requested real-source shards. Returns [(subset, parquet_file)]."""
    gm = json.load(open("checkpoints/cf_generator_map.json"))    # "subset/file" -> {gen: n}
    sm = json.load(open("checkpoints/cf_shard_map.json"))
    src = json.load(open("checkpoints/cf_real_sources.json"))    # small file -> [source, n]
    mb = {f"{k}/{e['file']}": e["mb"] for k in ("small", "eval") for e in sm[k] if "file" in e}

    fam_imgs = {sh: collections.Counter() for sh in gm}
    for sh, gens in gm.items():
        for g, n in gens.items():
            fam_imgs[sh][family(g)] += n

    chosen, used = [], set()
    for fam, tgt in targets.items():
        cand = sorted(((fam_imgs[sh][fam], -mb.get(sh, 9999), sh) for sh in gm
                       if fam_imgs[sh][fam] > 0 and sh not in used), reverse=True)
        got = 0
        for cnt, _, sh in cand:
            used.add(sh); chosen.append(sh); got += cnt
            if got >= tgt:
                break
    for source, n in real_shards:
        cand = sorted((mb[f"small/{f}"], f"small/{f}") for f, (s, _) in src.items()
                      if s == source and f"small/{f}" in mb)[:n]
        for _, sh in cand:
            if sh not in used:
                used.add(sh); chosen.append(sh)
    return [(sh.split("/")[0], sh.split("/", 1)[1]) for sh in chosen]


def select_by_generator(gen_targets):
    """Pick shards richest in each requested generator substring (case-insensitive)."""
    gm = json.load(open("checkpoints/cf_generator_map.json"))
    chosen, used = [], set()
    for sub, tgt in gen_targets.items():
        cand = []
        for sh, gens in gm.items():
            c = sum(n for g, n in gens.items() if sub.lower() in g.lower())
            if c > 0 and sh not in used:
                cand.append((c, sh))
        cand.sort(reverse=True)
        got = 0
        for c, sh in cand:
            used.add(sh); chosen.append(sh); got += c
            if got >= tgt:
                break
    return [(sh.split("/")[0], sh.split("/", 1)[1]) for sh in chosen]


@torch.no_grad()
def extract(models, subset, fname, batch=16):
    """Download a shard once; extract normalized feat7/feat12 for each backbone in
    `models` (name -> (model, proc, pool)). Same decoded images feed all backbones,
    so model_name stays aligned. Returns {name: (f7, f12)}, model_names."""
    tmp = tempfile.mkdtemp(dir="/tmp")  # local copy, fully removed after (no HF blob leak)
    path = hf_hub_download(REPO[subset], fname, repo_type="dataset", local_dir=tmp)
    d = pq.read_table(path, columns=["image_data", "model_name"]).to_pydict()
    acc = {n: {7: [], 12: []} for n in models}
    names = []
    for i in range(0, len(d["model_name"]), batch):
        imgs, mod = [], []
        for j in range(i, min(i + batch, len(d["model_name"]))):
            try:
                im = Image.open(io.BytesIO(d["image_data"][j])).convert("RGB")
            except Exception:
                continue
            if min(im.size) < 8:
                continue
            imgs.append(im); mod.append(d["model_name"][j])
        if not imgs:
            continue
        for n, (model, proc, pool) in models.items():
            f = B.features(model, proc, pool, n, imgs, DEVICE)
            acc[n][7].append(f[7]); acc[n][12].append(f[12])
        names += mod
    shutil.rmtree(tmp, ignore_errors=True)
    return {n: (torch.cat(acc[n][7]), torch.cat(acc[n][12])) for n in models}, names


def parse_targets(s):
    d = {"GAN": 1500, "PixelDiff": 2500, "LatentDiff": 9000, "Flow": 3500, "Commercial": 3000}
    if s:
        d = {k: int(v) for k, v in (kv.split("=") for kv in s.split(","))}
    return d


def main():
    ap = argparse.ArgumentParser(description="Extract CF shards to per-backbone caches")
    ap.add_argument("--targets", default="", help="family targets, e.g. Flow=4000")
    ap.add_argument("--generators", default="", help="generator-substring targets, e.g. flux=3200")
    ap.add_argument("--faces", type=int, default=2, help="FFHQ shards (each ~3k)")
    ap.add_argument("--backbones", default="", help="comma list; default the five ViT-Base models")
    args = ap.parse_args()
    bbs = args.backbones.split(",") if args.backbones else list(B.BASE5)

    if args.generators:
        gen_targets = {k: int(v) for k, v in (kv.split("=") for kv in args.generators.split(","))}
        targets = gen_targets
        shards = select_by_generator(gen_targets)
    else:
        targets = parse_targets(args.targets)
        real = [("FFHQ", args.faces), ("COCO", 1), ("LandscapesHQ", 1)]
        shards = select_shards(targets, real)
    for bb in bbs:
        os.makedirs(shard_dir(bb), exist_ok=True)

    def cached(bb, subset, fname):
        return os.path.exists(f"{shard_dir(bb)}/{subset}__{os.path.basename(fname)}.pt")

    todo = [(s, f) for s, f in shards if any(not cached(bb, s, f) for bb in bbs)]
    print(f"targets {targets} | backbones {bbs} | {len(shards)} shards, {len(todo)} to fetch | {DEVICE}")

    models = {bb: B.load(bb, DEVICE) for bb in bbs}
    for k, (subset, fname) in enumerate(todo, 1):
        need = {bb: models[bb] for bb in bbs if not cached(bb, subset, fname)}
        feats, mods = extract(need, subset, fname)
        for bb, (f7, f12) in feats.items():
            torch.save({"f7": f7, "f12": f12, "model_name": mods, "subset": subset, "file": fname},
                       f"{shard_dir(bb)}/{subset}__{os.path.basename(fname)}.pt")
        print(f"[{k}/{len(todo)}] {subset}/{os.path.basename(fname):<28} +{len(mods)} imgs "
              f"x{len(need)} backbones", flush=True)

    fams = collections.Counter()
    for f in glob.glob(f"{shard_dir(bbs[0])}/*.pt"):
        for g in torch.load(f, map_location="cpu", weights_only=False)["model_name"]:
            fams[family(g)] += 1
    print("\ncached family totals:", dict(fams))


if __name__ == "__main__":
    main()
