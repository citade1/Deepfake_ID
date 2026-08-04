"""Executable spec for CF's data semantics -- the two rules this project got wrong twice.
Run: python tests/test_cf_data.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cf_data import family, real_dataset

FAMILY_CASES = {
    # label=0 is real regardless of which generator the row is paired with
    ("MidjourneyV6_1", "Commercial", 0): "REAL",
    ("FLUX-dev", "Commercial", 0): "REAL",
    ("FFHQ", "Real", 0): "REAL",
    ("LandscapesHQ", "Real", 0): "REAL",
    # architecture drives the fake families
    ("BigGAN", "GAN", 1): "GAN",
    ("StyleSANXL", "GAN", 1): "GAN",
    ("glide", "PixDiff", 1): "PixDiff",
    ("DeepFloyd", "PixDiff", 1): "PixDiff",
    ("Imagen3", "Commercial", 1): "Commercial",
    ("Firefly_Image2", "Commercial", 1): "Commercial",
    ("gsdf/Counterfeit-V2.5", "LatDiff", 1): "LatDiff",
    # community models whose usernames used to trip the old regex
    ("pharmapsychotic/sugar-glider", "LatDiff", 1): "LatDiff",
    ("dataautogpt3/OpenDalle", "LatDiff", 1): "LatDiff",
    ("kukuhtw/ganjarpranowo", "LatDiff", 1): "LatDiff",
    ("Aditi2002/my-pet-dog", "LatDiff", 1): "LatDiff",
    # flow matching, split out by exact name (CF has no Flow architecture)
    ("FLUX-dev", "Commercial", 1): "Flow",
    ("FLUX-schnell", "Commercial", 1): "Flow",
    ("LFM", "Other", 1): "Flow",
    # CF's "Other" bucket is a family of its own, not folded into LatDiff
    ("tamingTransformers", "Other", 1): "Other",
    ("stable_cascade", "Other", 1): "Other",
}


def test_family():
    wrong = {k: (family(*k), exp) for k, exp in FAMILY_CASES.items() if family(*k) != exp}
    assert not wrong, f"misclassified: {wrong}"


def test_unknown_architecture_raises():
    """A new architecture value must fail loudly instead of being bucketed silently."""
    try:
        family("some/model", "BrandNewArch", 1)
    except ValueError:
        return
    raise AssertionError("unknown architecture should raise")


SOURCE_CASES = {
    # Small: the dataset is the model_name, capitalised
    ("FFHQ", "N/A"): "ffhq",
    ("COCO", "N/A"): "coco",
    ("LandscapesHQ", "N/A"): "landscapeshq",
    ("VISION", "N/A"): "vision",
    # Eval: model_name is the paired generator, the dataset is in real_source
    ("MidjourneyV6_1", "LAION"): "laion",
    ("DFGAN", "RAISE"): "raise",
    ("Hourglass", "ffhq"): "ffhq",
    ("BigGAN", "imagenet"): "imagenet",
    # datasets outside our lookup table are still captured, not dropped
    ("SomeGen", "celeba"): "celeba",
    # nothing identifiable
    ("SomeGen", "N/A"): "unknown",
    ("SomeGen", ""): "unknown",
    ("SomeGen", None): "unknown",
}


def test_real_dataset():
    wrong = {k: (real_dataset(*k), exp) for k, exp in SOURCE_CASES.items() if real_dataset(*k) != exp}
    assert not wrong, f"misidentified: {wrong}"


if __name__ == "__main__":
    test_family()
    test_unknown_architecture_raises()
    test_real_dataset()
    print(f"OK: {len(FAMILY_CASES)} family + {len(SOURCE_CASES)} real-source cases, "
          f"plus the unknown-architecture guard")
