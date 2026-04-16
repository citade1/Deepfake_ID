"""
DataLoader helpers for the Deepfake-vs-Real-60K HuggingFace dataset.

Label convention (after flipping):
    0 = Real
    1 = Fake
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import ViTImageProcessor

DATASET_STR = "prithivMLmods/Deepfake-vs-Real-60K"


def flip_labels(example: dict) -> dict:
    """
    The original dataset has 0=Fake, 1=Real.
    Flip so 0=Real, 1=Fake for conventional deepfake detection labelling.
    """
    example["label"] = 1 - example["label"]
    return example


def make_collate_fn(processor: ViTImageProcessor):
    """Return a collate function bound to the given processor."""

    def collate_fn(batch):
        images = []
        for item in batch:
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        inputs = processor(images=images, return_tensors="pt")
        return inputs, labels

    return collate_fn


def get_dataloaders(
    processor: ViTImageProcessor,
    dataset_str: str = DATASET_STR,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    Load the HuggingFace dataset, flip labels, and return DataLoaders.

    Returns:
        train_loader: DataLoader for the training split.
        test_loader:  DataLoader for the test split, or None if unavailable.
    """
    dataset = load_dataset(dataset_str)
    dataset = dataset.map(flip_labels)

    collate_fn = make_collate_fn(processor)
    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(dataset["train"], shuffle=True, **loader_kwargs)

    if "test" in dataset:
        test_loader = DataLoader(dataset["test"], shuffle=False, **loader_kwargs)
    else:
        test_loader = None

    return train_loader, test_loader
