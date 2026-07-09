"""DataLoaders for TheKernel01/Tiny-GenImage (label 0=Real, 1=Fake native, no
flip; per-image `generator` tag for cross-generator experiments)."""

import io

import torch
from datasets import load_dataset
from PIL import Image

DATASET_STR = "TheKernel01/Tiny-GenImage"
HF_REAL = 0  # label id for "Real"


class Collate:
    # Module-level (picklable) so num_workers>0 works with spawn on macOS.
    # jpeg_quality re-encodes as JPEG at that quality (robustness experiment).
    def __init__(self, processor, jpeg_quality: int = None):
        self.processor = processor
        self.jpeg_quality = jpeg_quality

    def _prep(self, item):
        im = item["image"]
        if im.mode != "RGB":
            im = im.convert("RGB")
        if self.jpeg_quality is not None:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=self.jpeg_quality)
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
        return im

    def __call__(self, batch):
        images = [self._prep(item) for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return self.processor(images=images, return_tensors="pt"), labels


def make_splits(
    dataset_str: str = DATASET_STR,
    ref_size: int = 1024,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    max_train: int = None,
    max_eval: int = None,
):
    """Return raw HF datasets (train_ds, val_ds, test_ds, ref_ds).

    Uses the dataset's own test split if present, otherwise carves one from
    train. ref_ds holds real-only images disjoint from train/val/test.
    max_train / max_eval optionally cap split sizes (memory-limited machines).
    """
    ds = load_dataset(dataset_str)

    if "test" in ds:
        train_full, test_ds = ds["train"], ds["test"]
    else:
        sp = ds["train"].train_test_split(
            test_size=test_frac, seed=seed, stratify_by_column="label")
        train_full, test_ds = sp["train"], sp["test"]

    real_idx = [i for i, lab in enumerate(train_full["label"]) if lab == HF_REAL]
    ref_idx = set(real_idx[:ref_size])
    ref_ds = train_full.select(sorted(ref_idx))
    query_ds = train_full.select(sorted(set(range(len(train_full))) - ref_idx))

    sp = query_ds.train_test_split(
        test_size=val_frac, seed=seed, stratify_by_column="label")
    train_ds, val_ds = sp["train"], sp["test"]

    def cap(d, n):
        return d.shuffle(seed=seed).select(range(min(n, len(d)))) if n else d

    return cap(train_ds, max_train), cap(val_ds, max_eval), cap(test_ds, max_eval), ref_ds
