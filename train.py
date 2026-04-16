"""
Training script: 1-epoch fine-tuning of the Multi-Layer LID deepfake classifier.

Pipeline:
    1. Load frozen ViT backbone from "prithivMLmods/Deep-Fake-Detector-v2-Model".
    2. Build MultiLayerLIDModel (only the MLP classifier is trainable).
    3. Train for 1 epoch on "prithivMLmods/Deepfake-vs-Real-60K".
    4. Evaluate on the test split (accuracy + AUC).
    5. Save the classifier weights to ./model/lid_classifier.pth
       (the ViT backbone is loaded from HuggingFace at inference time).

Label convention: 0 = Real, 1 = Fake.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

from utils.dataloader_helper import get_dataloaders
from utils.lid_estimator import twonn_global_id
from utils.model import MultiLayerLIDModel

# ── Config ───────────────────────────────────────────────────────────────────
PRETRAINED  = "prithivMLmods/Deep-Fake-Detector-v2-Model"
SAVE_DIR    = "./model"
SAVE_PATH   = os.path.join(SAVE_DIR, "lid_classifier.pth")
BATCH_SIZE  = 32
LR          = 1e-3
K           = 20            # k-NN for LID features (must be < batch_size)
NUM_WORKERS = 4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(model: MultiLayerLIDModel, loader, device: str):
    """Return (accuracy, AUC) on the given DataLoader."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            pv = inputs["pixel_values"].to(device)
            logits = model(pv)
            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return acc, auc


def train():
    print(f"Device : {DEVICE}")
    print(f"Config : batch={BATCH_SIZE}, k={K}, lr={LR}")

    # ── Load backbone & processor ────────────────────────────────────────────
    print(f"\nLoading pre-trained ViT from '{PRETRAINED}' …")
    processor  = ViTImageProcessor.from_pretrained(PRETRAINED)
    base_model = ViTForImageClassification.from_pretrained(PRETRAINED)

    # ── Build model ──────────────────────────────────────────────────────────
    model = MultiLayerLIDModel(base_model, k=K).to(DEVICE)
    probe_names = list(model.probe_indices.keys())
    probe_idxs  = list(model.probe_indices.values())
    n_trainable = sum(p.numel() for p in model.trainable_parameters())

    print(f"Probing ViT encoder layers : {dict(zip(probe_names, probe_idxs))}")
    print(f"Trainable parameters       : {n_trainable:,}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    print("\nLoading dataset …")
    train_loader, test_loader = get_dataloaders(
        processor,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # ── Optimizer (classifier only) ──────────────────────────────────────────
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=LR)

    # ── 1 Epoch training ─────────────────────────────────────────────────────
    print("\n── Training (1 epoch) ───────────────────────────────────────────")
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    log_interval = 50   # print every N steps

    for step, (inputs, labels) in enumerate(
        tqdm(train_loader, desc="Training")
    ):
        pv     = inputs["pixel_values"].to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(pv)
        loss   = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct      += (logits.argmax(1) == labels).sum().item()
        total        += labels.size(0)

        if step % log_interval == 0:
            # Log global LID per layer every log_interval steps (no grad needed)
            lid_info = model.analyze_lid(pv)
            lid_str  = "  ".join(
                f"{n}={v:.2f}" for n, v in lid_info.items()
            )
            print(
                f"  step {step:5d} | loss {loss.item():.4f} "
                f"| acc {correct / total:.4f} | LID {lid_str}"
            )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc  = correct / total
    print(f"\nEpoch done — avg loss: {epoch_loss:.4f}  |  train acc: {epoch_acc:.4f}")

    # ── Evaluation ───────────────────────────────────────────────────────────
    if test_loader is not None:
        print("\n── Evaluation ───────────────────────────────────────────────────")
        acc, auc = evaluate(model, test_loader, DEVICE)
        print(f"Test  —  accuracy: {acc:.4f}  |  AUC: {auc:.4f}")
    else:
        print("\nNo test split found — skipping evaluation.")

    # ── Save classifier weights ───────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.classifier.state_dict(), SAVE_PATH)
    print(f"\nClassifier weights saved → {SAVE_PATH}")


if __name__ == "__main__":
    train()
