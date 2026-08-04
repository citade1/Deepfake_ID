"""Download CF shards once and extract per-shard features for every backbone. Resumable."""
import argparse
import collections
import concurrent.futures as cf
import glob
import io
import json
import os
import shutil
import tempfile
import time
import urllib.request

import pyarrow.parquet as pq
import torch
from PIL import Image

from utils import backbones as B
from utils import paths as P
from pipeline.compose import DEFAULT_FAKE
from utils.cf_data import REPO, family

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# floor on distinct generators per family
MIN_GENERATORS = {"GAN": 14, "PixDiff": 4, "LatDiff": 150, "Flow": 3, "Commercial": 9, "Other": 2}

# extract this many times what compose draws, so a --seed has room to pick a different subset
DRAW_MULTIPLE = 2

# images to pull per real source, at DRAW_MULTIPLE x what compose draws from each
REAL_TARGET = {"ffhq": 7000, "coco": 3500, "landscapeshq": 3500}

JPEG_QUALITY, JPEG_SUBSAMPLING = 96, 0
PREPROC = f"jpeg_q{JPEG_QUALITY}_s{JPEG_SUBSAMPLING}"


def _load_maps():
    """-> by_family[shard][family]={gen: n} over fakes, mb[shard], src[shard]={source: n},
    and totals[family]={gen: n} over the whole corpus."""
    gm = json.load(open(P.GENERATOR_MAP))
    sm = json.load(open(P.SHARD_MAP))
    src = json.load(open(P.REAL_SOURCE_MAP))
    mb = {f"{k}/{e['file']}": e["mb"] for k in ("small", "eval") for e in sm[k]}
    by_family, totals = {}, collections.defaultdict(collections.Counter)
    for sh, gens in gm.items():
        fams = collections.defaultdict(collections.Counter)
        for key, n in gens.items():                  # key is "model_name\tarchitecture"
            m, a = key.split("\t")
            fams[family(m, a, 1)][m] += n
        by_family[sh] = fams
        for f, c in fams.items():
            totals[f].update(c)
    return by_family, mb, src, totals


def _pick_fake(by_family, mb, fam, target, min_gens, used, totals):
    """Shards for one family; a generator counts only once it holds its share of the draw."""
    quota = -(-target // min_gens) if min_gens else 0
    need = {g: min(quota, n) for g, n in totals[fam].items()}   # some generators are just small
    have = collections.Counter()
    for sh in used:
        have.update(by_family[sh].get(fam, {}))
    covered = lambda: sum(have[g] >= need[g] for g in need)

    picked = []
    while not (sum(have.values()) >= target and covered() >= min_gens):
        best, best_key = None, None
        for sh, fams in by_family.items():
            if sh in used or fam not in fams:
                continue
            gain = sum(min(n, max(0, need.get(g, 0) - have[g])) for g, n in fams[fam].items())
            key = (gain, sum(fams[fam].values()), -mb.get(sh, 9999))
            if best_key is None or key > best_key:
                best, best_key = sh, key
        if best is None or best_key[:2] == (0, 0):
            break
        used.add(best); picked.append(best)
        have.update(by_family[best][fam])
    print(f"  {fam:>12}: {sum(have.values()):>6} imgs, "
          f"{covered():>3}/{len(totals[fam])} generators at quota {quota}")
    return picked


def _pick_real(src, mb, source, target, used):
    """Cheapest shards, by megabytes per image, until `source` reaches `target`. -> [shard]"""
    cand = sorted((mb[sh] / counts[source], sh) for sh, counts in src.items()
                  if source in counts and sh in mb and sh not in used)
    have, picked = 0, []
    for _, sh in cand:
        if have >= target:
            break
        used.add(sh); picked.append(sh); have += src[sh][source]
    return picked


def _split(shard):
    """"small/data/x.parquet" -> ("small", "data/x.parquet")"""
    return tuple(shard.split("/", 1))


def select_shards(targets, real_shards, min_generators=None):
    """Shards meeting every family and real-source target. -> [(subset, file)]."""
    by_family, mb, src, totals = _load_maps()
    min_generators = min_generators or {}
    chosen, used = [], set()
    for fam, tgt in targets.items():
        chosen += _pick_fake(by_family, mb, fam, tgt, min_generators.get(fam, 0), used, totals)
    for source, tgt in real_shards:
        chosen += _pick_real(src, mb, source, tgt, used)
    return [_split(sh) for sh in chosen]


def select_by_generator(gen_targets):
    """Pick shards richest in each requested generator substring (case-insensitive)."""
    by_family, _, _, _ = _load_maps()
    chosen, used = [], set()
    for sub, tgt in gen_targets.items():
        cand = []
        for sh, fams in by_family.items():
            c = sum(n for gens in fams.values() for g, n in gens.items()
                    if sub.lower() in g.lower())
            if c > 0 and sh not in used:
                cand.append((c, sh))
        cand.sort(reverse=True)
        got = 0
        for c, sh in cand:
            used.add(sh); chosen.append(sh); got += c
            if got >= tgt:
                break
    return [_split(sh) for sh in chosen]


def _fetch(subset, fname, workers=4, chunk=16 << 20):
    """Download one shard into its own temp dir using parallel range requests."""
    url = f"https://huggingface.co/datasets/{REPO[subset]}/resolve/main/{fname}"
    head = urllib.request.Request(url, method="HEAD")      # a GET would leave a body to drain
    with urllib.request.urlopen(head, timeout=60) as r:
        total = int(r.headers.get("x-linked-size") or r.headers["Content-Length"])
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, os.path.basename(fname))
    spans = [(i, min(i + chunk, total) - 1) for i in range(0, total, chunk)]

    def part(span):
        req = urllib.request.Request(url, headers={"Range": f"bytes={span[0]}-{span[1]}"})
        for attempt in range(4):
            try:
                return span[0], urllib.request.urlopen(req, timeout=180).read()
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    with open(path, "wb") as fh:
        fh.truncate(total)
        with cf.ThreadPoolExecutor(workers) as ex:
            for f in cf.as_completed([ex.submit(part, s) for s in spans]):
                off, buf = f.result()
                fh.seek(off); fh.write(buf)
    if os.path.getsize(path) != total:
        raise IOError(f"{fname}: got {os.path.getsize(path)} of {total} bytes")
    return tmp, path


def _requantize(im):
    """One compression history for every image, so the class cannot be read off the format."""
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@torch.no_grad()
def extract(models, path, batch=16):
    """Extract the designated features for every backbone from one downloaded shard."""
    acc = {n: {7: [], 12: []} for n in models}
    meta = {"model_name": [], "architecture": [], "label": [], "real_source": []}
    cols = ["image_data", "model_name", "architecture", "label", "real_source"]
    # a Small shard is one 2GB row group, so read_table would hold every image at once
    for rb in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=cols):
        d = rb.to_pydict()
        imgs, keep = [], []
        for j in range(len(d["model_name"])):
            try:
                im = Image.open(io.BytesIO(d["image_data"][j])).convert("RGB")
            except Exception:
                continue
            if min(im.size) < 8:
                continue
            im = _requantize(im)
            imgs.append(im); keep.append(j)
        if not imgs:
            continue
        for n, (model, proc, pool) in models.items():
            f = B.features(model, proc, pool, n, imgs, DEVICE)
            acc[n][7].append(f[7]); acc[n][12].append(f[12])
        for k in meta:
            meta[k] += [d[k][j] for j in keep]
    if not meta["label"]:                  # every image in the shard failed to decode
        empty = torch.empty(0, 0)
        return {n: (empty, empty) for n in models}, meta
    return {n: (torch.cat(acc[n][7]), torch.cat(acc[n][12])) for n in models}, meta


def parse_targets(s):
    if s:
        return {k: int(v) for k, v in (kv.split("=") for kv in s.split(","))}
    return {f: n * DRAW_MULTIPLE for f, n in DEFAULT_FAKE.items()}


def main():
    ap = argparse.ArgumentParser(description="Extract CF shards to per-backbone caches")
    ap.add_argument("--targets", default="", help="family targets, e.g. Flow=4000")
    ap.add_argument("--generators", default="", help="generator-substring targets, e.g. flux=3200")
    ap.add_argument("--real-target", default=",".join(f"{k}={v}" for k, v in REAL_TARGET.items()),
                    help="images to pull per real source, e.g. ffhq=7000")
    ap.add_argument("--backbones", default="", help="comma list; default the five ViT-Base models")
    ap.add_argument("--min-generators",
                    default=",".join(f"{k}={v}" for k, v in MIN_GENERATORS.items()),
                    help="minimum distinct generators to cover per family")
    args = ap.parse_args()
    bbs = args.backbones.split(",") if args.backbones else list(B.BASE5)

    if args.generators:
        gen_targets = {k: int(v) for k, v in (kv.split("=") for kv in args.generators.split(","))}
        targets = gen_targets
        shards = select_by_generator(gen_targets)
    else:
        targets = parse_targets(args.targets)
        min_gens = {k: int(v) for k, v in (kv.split("=") for kv in args.min_generators.split(","))}
        real = [(k, int(v)) for k, v in (kv.split("=") for kv in args.real_target.split(","))]
        shards = select_shards(targets, real, min_gens)
    for bb in bbs:
        os.makedirs(P.shards_dir(bb), exist_ok=True)

    def cached(bb, subset, fname):
        return os.path.exists(f"{P.shards_dir(bb)}/{subset}__{os.path.basename(fname)}.pt")

    todo = [(s, f) for s, f in shards if any(not cached(bb, s, f) for bb in bbs)]
    print(f"targets {targets} | backbones {bbs} | {len(shards)} shards, {len(todo)} to fetch | {DEVICE}")

    models = {bb: B.load(bb, DEVICE) for bb in bbs} if todo else {}
    # fetch the next shard while this one goes through the backbones
    pool = cf.ThreadPoolExecutor(1)
    ahead = pool.submit(_fetch, *todo[0]) if todo else None
    for k, (subset, fname) in enumerate(todo, 1):
        tmp, path = ahead.result()
        ahead = pool.submit(_fetch, *todo[k]) if k < len(todo) else None
        need = {bb: models[bb] for bb in bbs if not cached(bb, subset, fname)}
        try:
            feats, meta = extract(need, path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        for bb, (f7, f12) in feats.items():
            torch.save({"f7": f7, "f12": f12, **meta, "subset": subset, "file": fname,
                        "preproc": PREPROC},
                       f"{P.shards_dir(bb)}/{subset}__{os.path.basename(fname)}.pt")
        print(f"[{k}/{len(todo)}] {subset}/{os.path.basename(fname):<28} "
              f"+{len(meta['label'])} imgs x{len(need)} backbones", flush=True)
    pool.shutdown()

    fams, gens = collections.Counter(), collections.defaultdict(set)
    for f in glob.glob(f"{P.shards_dir(bbs[0])}/*.pt"):   
        b = torch.load(f, map_location="cpu", weights_only=False, mmap=True) # mmap: only the labels are read
        for m, a, l in zip(b["model_name"], b["architecture"], b["label"]):
            fm = family(m, a, l)
            fams[fm] += 1
            if fm != "REAL":
                gens[fm].add(m)
    print("\ncached image totals:", dict(fams))
    print("cached generators per family:", {k: len(v) for k, v in gens.items()})


if __name__ == "__main__":
    main()
