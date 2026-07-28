"""Unit test for family() -- generator names are noisy HuggingFace repo IDs, so
guard against username substring false positives (e.g. '...ganjar...', '...aadith...').
Run: python -m pytest tests/  or  python tests/test_family.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cf_data import family

CASES = {
    # real generators that actually appear in the composed data
    "StyleSANXL": "GAN", "GigaGAN": "GAN", "DFGAN": "GAN",
    "BigGAN": "GAN", "StyleGAN2": "GAN", "ProjectedGAN": "GAN",
    "glide": "PixelDiff", "DiT": "PixelDiff", "vqdiffusion": "PixelDiff",
    "FLUX-schnell": "Flow", "LFM": "Flow", "FLUX-dev": "Flow",
    "Firefly_Image2": "Commercial", "Imagen3": "Commercial", "IdeogramV2": "Commercial",
    "Yntec/DreamShaperRemix": "LatentDiff", "stablediffusionapi/divineelegancemix": "LatentDiff",
    "COCO": "REAL", "FFHQ": "REAL",
    # username false positives that must NOT be mistaken for GAN / PixelDiff
    "kukuhtw/ganjarpranowo": "LatentDiff", "simbolo-ai/bagan": "LatentDiff",
    "Gangotri2205/my-pet-dog": "LatentDiff", "AadithKumar/my-pet-dog-eaak": "LatentDiff",
    "Aditi2002/my-pet-dog": "LatentDiff", "Aditya1911/random-image-qwe": "LatentDiff",
}


def test_family():
    wrong = {g: (family(g), exp) for g, exp in CASES.items() if family(g) != exp}
    assert not wrong, f"misclassified: {wrong}"


if __name__ == "__main__":
    test_family()
    print(f"OK: {len(CASES)} generator names classified correctly")
