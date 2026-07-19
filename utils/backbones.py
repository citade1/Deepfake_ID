"""The five study backbones (all ViT-Base, 768-d) and a unified feature extractor,
so the geometry analyses run identically across diverse training objectives."""
import torch
import torch.nn.functional as F
from transformers import (AutoImageProcessor, AutoModel, CLIPImageProcessor,
                          CLIPVisionModel, ViTMAEModel, ViTModel)

# name -> (hf id, pooling): cls token, or mean over patches (SigLIP has no cls)
BACKBONES = {
    "clip":    ("openai/clip-vit-base-patch16", "cls"),      # language contrastive
    "siglip2": ("google/siglip2-base-patch16-224", "mean"),  # sigmoid language contrastive
    "dinov2":  ("facebook/dinov2-base", "cls"),              # self-supervised distillation
    "mae":     ("facebook/vit-mae-base", "cls"),             # self-supervised masked reconstruction
    "vit":     ("google/vit-base-patch16-224", "cls"),       # pure supervised (ImageNet)
    "clipL":   ("openai/clip-vit-large-patch14", "cls"),     # CLIP L/14 headline (1024-d)
}

# the five ViT-Base backbones loaded together (clipL is a single-model headline run)
BASE5 = ["clip", "siglip2", "dinov2", "mae", "vit"]


def load(name, device):
    mid, pool = BACKBONES[name]
    if name == "clip":
        model = CLIPVisionModel.from_pretrained(mid)
        proc = CLIPImageProcessor.from_pretrained(mid)
    elif name == "mae":
        model = ViTMAEModel.from_pretrained(mid)
        model.config.mask_ratio = 0.0                        # keep all patches (no pretrain masking)
        proc = AutoImageProcessor.from_pretrained(mid)
    elif name == "vit":
        model = ViTModel.from_pretrained(mid)
        proc = AutoImageProcessor.from_pretrained(mid)
    else:                                                    # dinov2, siglip2
        model = AutoModel.from_pretrained(mid)
        proc = AutoImageProcessor.from_pretrained(mid)
    return model.to(device).eval(), proc, pool


def _vision(model, name):
    return model.vision_model if name == "siglip2" else model


@torch.no_grad()
def features(model, proc, pool, name, images, device, layers=(7, 12)):
    """Normalized ViT features at the given layers for a list of PIL images."""
    px = proc(images=images, return_tensors="pt")["pixel_values"].to(device)
    hs = _vision(model, name)(pixel_values=px, output_hidden_states=True).hidden_states
    out = {}
    for L in layers:
        h = hs[L][:, 0] if pool == "cls" else hs[L].mean(1)
        out[L] = F.normalize(h, dim=1).cpu()
    return out
