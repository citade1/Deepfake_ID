"""Bootstrap the shard-selection maps (checkpoints/cf_*.json) by scanning only the
model_name column of each CF parquet remotely -- no image download."""
import argparse
import collections
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = {"small": "OwensLab/CommunityForensics-Small", "eval": "OwensLab/CommunityForensics-Eval"}
REAL_SOURCES = {"COCO", "LandscapesHQ", "FFHQ", "VISION"}
OUT = "checkpoints"


def shard_model_names(fs, repo, f):
    with fs.open(f"datasets/{repo}/{f}") as fh:
        return pq.read_table(fh, columns=["model_name"]).column("model_name").to_pylist()


def main():
    ap = argparse.ArgumentParser(description="Rebuild prepare_cf's shard-selection maps")
    ap.add_argument("--limit", type=int, default=0, help="scan only N shards per subset (smoke test)")
    args = ap.parse_args()
    api, fs = HfApi(), HfFileSystem()
    gen_map, shard_map, real_src = {}, {"small": [], "eval": []}, {}

    for subset, repo in REPO.items():
        files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
        files = sorted(files)[:args.limit] if args.limit else sorted(files)
        sizes = {i.path: i.size for i in api.get_paths_info(repo, files, repo_type="dataset")}
        for k, f in enumerate(files, 1):
            names = shard_model_names(fs, repo, f)
            counts = collections.Counter(names)
            gen_map[f"{subset}/{f}"] = dict(counts)
            shard_map[subset].append({"file": f, "mb": round((sizes.get(f) or 0) / 1e6), "rows": len(names)})
            reals = {s: n for s, n in counts.items() if s in REAL_SOURCES}
            if subset == "small" and reals:
                src, n = max(reals.items(), key=lambda kv: kv[1])
                real_src[f] = [src, n]
            print(f"[{subset} {k}/{len(files)}] {f}  {len(counts)} generators", flush=True)

    os.makedirs(OUT, exist_ok=True)
    json.dump(gen_map, open(f"{OUT}/cf_generator_map.json", "w"))
    json.dump(shard_map, open(f"{OUT}/cf_shard_map.json", "w"))
    json.dump(real_src, open(f"{OUT}/cf_real_sources.json", "w"))
    print(f"wrote maps: {len(gen_map)} shards, {len(real_src)} real shards")


if __name__ == "__main__":
    main()
