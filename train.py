"""Train the Multi-Layer LID deepfake classifier.

Builds a frozen real-image reference bank, trains the MLP head with early
stopping on validation AUC, evaluates on the test split, and saves the
classifier + reference bank. Quick smoke test: python train.py --epochs 1.
See README.
"""

import argparse
import copy
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

from utils.dataloader_helper import get_dataloaders
from utils.model import MultiLayerLIDModel

PRETRAINED  = "prithivMLmods/Deep-Fake-Detector-v2-Model"
SAVE_DIR    = "./checkpoints"
SAVE_PATH   = os.path.join(SAVE_DIR, "lid_classifier.pth")
BATCH_SIZE  = 32
LR          = 1e-3
K           = 20
REF_SIZE    = 1024
EPOCHS      = 10
PATIENCE    = 3
NUM_WORKERS = 4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels_all, probs_all = [], [], []
    for inputs, labels in tqdm(loader, desc="Eval", leave=False):
        logits = model(inputs["pixel_values"].to(device))
        probs_all.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.numpy())
    acc = accuracy_score(labels_all, preds)
    try:
        auc = roc_auc_score(labels_all, probs_all)
    except ValueError:
        auc = float("nan")
    return acc, auc


def train(epochs=EPOCHS):
    print(f"Device: {DEVICE} | batch={BATCH_SIZE} k={K} lr={LR} "
          f"epochs={epochs} patience={PATIENCE} ref_size={REF_SIZE}")

    processor  = ViTImageProcessor.from_pretrained(PRETRAINED)
    base_model = ViTForImageClassification.from_pretrained(PRETRAINED)
    model      = MultiLayerLIDModel(base_model, k=K).to(DEVICE)
    print(f"Trainable params: {sum(p.numel() for p in model.trainable_parameters()):,}")

    train_loader, val_loader, test_loader, ref_loader = get_dataloaders(
        processor, batch_size=BATCH_SIZE, ref_size=REF_SIZE, num_workers=NUM_WORKERS)

    print("Building reference bank …")
    print("Reference shapes:", model.build_reference_bank(ref_loader, DEVICE, max_size=REF_SIZE))

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=LR)
    best_auc, best_state, stale = -1.0, copy.deepcopy(model.classifier.state_dict()), 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Train e{epoch}"):
            labels = labels.to(DEVICE)
            logits = model(inputs["pixel_values"].to(DEVICE))
            loss   = F.cross_entropy(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)
        print(f"Epoch {epoch}: loss {loss_sum/max(1,len(train_loader)):.4f} "
              f"acc {correct/max(1,total):.4f}")

        val_acc, val_auc = evaluate(model, val_loader, DEVICE)
        print(f"  val acc {val_acc:.4f} auc {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, stale = val_auc, 0
            best_state = copy.deepcopy(model.classifier.state_dict())
            print(f"  ✓ best val auc {best_auc:.4f}")
        else:
            stale += 1
            if stale >= PATIENCE:
                print("Early stopping."); break

    model.classifier.load_state_dict(best_state)
    test_acc, test_auc = evaluate(model, test_loader, DEVICE)
    print(f"\nTest — acc {test_acc:.4f} auc {test_auc:.4f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save({
        "classifier": best_state,
        "reference_bank": {n: v.cpu() for n, v in model.reference_bank().items()},
        "k": K,
        "probe_indices": model.probe_indices,
        "best_val_auc": best_auc,
    }, SAVE_PATH)
    print(f"Saved → {SAVE_PATH} (best val auc {best_auc:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-layer LID training")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Max epochs (set 1 for a quick smoke test).")
    args, _ = parser.parse_known_args()
    train(epochs=args.epochs)
