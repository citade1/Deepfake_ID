"""DataLoaders for the HF dataset prithivMLmods/Deepfake-vs-Real-60K (gated).

HF labels {0: Fake, 1: Real} are flipped to our convention 0=Real, 1=Fake.
The reference bank is a disjoint subset of real images (never used as a query),
so query LID has no distance-0 self match. See README.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor

DATASET_STR = "prithivMLmods/Deepfake-vs-Real-60K"
HF_REAL = 1  # raw HF label id for "Real"


def make_collate_fn(processor: ViTImageProcessor):
    def collate_fn(batch):
        images = [
            item["image"] if item["image"].mode == "RGB" else item["image"].convert("RGB")
            for item in batch
        ]
        labels = torch.tensor([1 - item["label"] for item in batch], dtype=torch.long)
        return processor(images=images, return_tensors="pt"), labels

    return collate_fn


def get_dataloaders(
    processor: ViTImageProcessor,
    dataset_str: str = DATASET_STR,
    batch_size: int = 32,
    ref_size: int = 1024,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
):
    """Return (train_loader, val_loader, test_loader, ref_loader).

    Uses the dataset's own test split if present, otherwise carves one from
    train. ref_loader holds real-only images disjoint from train/val/test.
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

    collate_fn = make_collate_fn(processor)
    common = dict(batch_size=batch_size, collate_fn=collate_fn,
                  num_workers=num_workers, pin_memory=torch.cuda.is_available())

    print(f"Splits — train:{len(train_ds)} val:{len(val_ds)} "
          f"test:{len(test_ds)} ref(real):{len(ref_ds)}")
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
        DataLoader(ref_ds, shuffle=False, **common),
    )
