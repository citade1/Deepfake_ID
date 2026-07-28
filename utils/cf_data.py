"""Load a composed Community Forensics dataset draw, and map a generator's
`model_name` to its family (see compose_cf.py)."""
import re

import torch

# generator families ordered oldest -> newest, for generational experiments
GENERATIONS = ["GAN", "PixelDiff", "LatentDiff", "Flow", "Commercial"]
REAL_SOURCES = {"COCO", "LandscapesHQ", "FFHQ", "VISION"}

# CF generator names are noisy HuggingFace repo IDs, so match architecture tokens
# specifically enough to avoid username false positives ("...ganjar...", "...aadith...").
# Short/ambiguous tokens (cips, dit) use word boundaries; the rest are distinctive.
_GAN = re.compile(r"stylegan|biggan|progan|gigagan|stylesan|dfgan|projectedgan"
                  r"|gansformer|styleswin|galip|hourglass|\bcips\b", re.I)
_PIXEL = re.compile(r"glide|guideddiffusion|vqdiffusion|deepfloyd|taming|\bdit\b", re.I)
_COMMERCIAL = re.compile(r"midjourney|dalle|imagen|firefly|ideogram", re.I)


def family(gen):
    """Map a model_name to a generator family; unmatched community models default
    to LatentDiff (Community Forensics is dominated by Stable-Diffusion fine-tunes)."""
    if gen in REAL_SOURCES:
        return "REAL"
    g = gen.lower()
    if _COMMERCIAL.search(g):
        return "Commercial"
    if "flux" in g or g == "lfm":
        return "Flow"
    if _GAN.search(g):
        return "GAN"
    if _PIXEL.search(g):
        return "PixelDiff"
    return "LatentDiff"


def load(seed=0, backbone="clip"):
    d = torch.load(f"checkpoints/cf_cache/dataset_{backbone}_s{seed}.pt",
                   map_location="cpu", weights_only=False)
    d["family"] = list(d["family"])
    d["generator"] = list(d["generator"])
    return d
