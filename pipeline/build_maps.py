"""Build the shard-selection maps from CF parquet metadata; no images are downloaded."""
import argparse
import collections
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from utils import paths as P
from utils.cf_data import REPO, real_dataset


def shard_rows(fs, repo, f):
    """Per-row (model_name, architecture, label, real_source)."""
    with fs.open(f"datasets/{repo}/{f}") as fh:
        t = pq.read_table(fh, columns=["model_name", "architecture", "label", "real_source"])
    return list(zip(t.column("model_name").to_pylist(),
                    t.column("architecture").to_pylist(),
                    t.column("label").to_pylist(),
                    t.column("real_source").to_pylist()))


def main():
    ap = argparse.ArgumentParser(description="Rebuild extract.py's shard-selection maps")
    ap.add_argument("--limit", type=int, default=0, help="scan only N shards per subset (smoke test)")
    ap.add_argument("--fold-only", action="store_true",
                    help="rebuild the maps from data/maps/scan.jsonl without touching the hub")
    args = ap.parse_args()

    # one JSON line per shard, so a dropped connection costs one shard
    P.ensure(P.MAPS)
    expected = 0
    scanned = {}
    if os.path.exists(P.SCAN_LOG):
        with open(P.SCAN_LOG) as fh:
            for line in fh:
                r = json.loads(line)
                scanned[r["shard"]] = r
        print(f"resuming: {len(scanned)} shards already scanned")

    if args.fold_only:
        print(f"fold-only: rebuilding maps from {len(scanned)} logged shards")
        return fold(scanned, len(scanned))

    api, fs = HfApi(), HfFileSystem()
    log = open(P.SCAN_LOG, "a")
    try:
        for subset, repo in REPO.items():
            files = [f for f in api.list_repo_files(repo, repo_type="dataset")
                     if f.endswith(".parquet")]
            files = sorted(files)[:args.limit] if args.limit else sorted(files)
            expected += len(files)
            sizes = {i.path: i.size for i in api.get_paths_info(repo, files, repo_type="dataset")}
            for k, f in enumerate(files, 1):
                key = f"{subset}/{f}"
                if key in scanned:
                    continue
                rows = shard_rows(fs, repo, f)
                # fake generators keyed by name+architecture; reals keyed by source dataset
                fakes = collections.Counter((m, a) for m, a, l, _ in rows if l == 1)
                reals = collections.Counter(real_dataset(m, rs) for m, _, l, rs in rows if l == 0)
                rec = {"shard": key,
                       "gens": {f"{m}\t{a}": n for (m, a), n in fakes.items()},
                       "mb": round((sizes.get(f) or 0) / 1e6),
                       "rows": len(rows),
                       "reals": dict(reals),
                       "arch": dict(collections.Counter(a for _, a, _, _ in rows))}
                scanned[key] = rec
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"[{subset} {k}/{len(files)}] {f}  {len(fakes)} fake generators, "
                      f"{sum(reals.values())} real", flush=True)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        # a dropped connection must not leave stale maps next to a good scan log
        print(f"\nscan interrupted ({type(e).__name__}: {e}) -- folding what was scanned")
    finally:
        log.close()
    fold(scanned, expected)


def fold(scanned, expected):
    """Write the three maps from the per-shard scan log."""
    gen_map, shard_map, real_src = {}, {"small": [], "eval": []}, {}
    arch_seen = collections.Counter()
    for key, r in scanned.items():
        subset, f = key.split("/", 1)
        gen_map[key] = r["gens"]
        shard_map[subset].append({"file": f, "mb": r["mb"], "rows": r["rows"],
                                  "n_real": sum(r["reals"].values())})
        arch_seen.update(r["arch"])
        # keep every real source: one shard can mix e.g. ffhq and vision
        if r["reals"]:
            real_src[key] = r["reals"]

    json.dump(gen_map, open(P.GENERATOR_MAP, "w"))
    json.dump(shard_map, open(P.SHARD_MAP, "w"))
    json.dump(real_src, open(P.REAL_SOURCE_MAP, "w"))
    print(f"\nwrote maps: {len(gen_map)} shards, {len(real_src)} shards with real images")
    print("architecture values across the whole corpus:", dict(arch_seen))
    if len(scanned) < expected:
        # a partial scan under-counts rare families and skews selection
        print(f"\nWARNING: {len(scanned)}/{expected} shards scanned -- maps are INCOMPLETE. "
              f"Re-run to resume before trusting family counts.")


if __name__ == "__main__":
    main()
