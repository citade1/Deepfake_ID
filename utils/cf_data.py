"""Load a composed Community Forensics dataset draw (see compose_cf.py)."""
import torch

# generator families ordered oldest -> newest, for generational experiments
GENERATIONS = ["GAN", "PixelDiff", "LatentDiff", "Flow", "Commercial"]


def load(seed=0, backbone="clip"):
    d = torch.load(f"checkpoints/cf_cache/dataset_{backbone}_s{seed}.pt",
                   map_location="cpu", weights_only=False)
    d["family"] = list(d["family"])
    d["generator"] = list(d["generator"])
    return d
