"""Every on-disk location in one place: maps/ (shard metadata), features/ (per backbone),
datasets/ (draws), images/ (raw pools)."""
import os

DATA = "data"
MAPS = f"{DATA}/maps"
FEATURES = f"{DATA}/features"
DATASETS = f"{DATA}/datasets"
IMAGES = f"{DATA}/images"

GENERATOR_MAP = f"{MAPS}/generators.json"
SHARD_MAP = f"{MAPS}/shards.json"
REAL_SOURCE_MAP = f"{MAPS}/real_sources.json"
SCAN_LOG = f"{MAPS}/scan.jsonl"   # one line per scanned shard, makes build_maps resumable


def shards_dir(backbone):
    return f"{FEATURES}/{backbone}/shards"


def pool_file(backbone):
    return f"{FEATURES}/{backbone}/pool.pt"


def dataset_file(backbone, seed):
    return f"{DATASETS}/{backbone}_s{seed}.pt"


def image_pool_file(tag):
    return f"{IMAGES}/pool_{tag}.pt"


def ensure(path):
    """Create the directory holding `path` (or `path` itself if it has no suffix)."""
    d = path if not os.path.splitext(path)[1] else os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    return path
