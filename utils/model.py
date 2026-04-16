"""
Multi-layer LID deepfake classifier.

Architecture:
    Frozen ViT backbone  →  CLS token extracted at start / middle / end layers
    →  per-layer LID feature vectors (TwoNN-style)
    →  concatenated (B, 3k) features
    →  shallow MLP classifier  →  2-class logits (real / fake)

The ViT is never updated; only the MLP classifier is trained.
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification

from .lid_estimator import compute_lid_features, twonn_global_id


class MultiLayerLIDModel(nn.Module):
    """
    Deepfake / Real binary classifier based on Local Intrinsic Dimension (LID)
    measured at three depths of a frozen ViT encoder.

    Args:
        vit_model:  A loaded ``ViTForImageClassification`` instance.
        k:          Number of nearest neighbours for LID feature computation.
                    Effective k is clamped to min(k, batch_size - 1) at
                    runtime so small batches are handled gracefully.

    Label convention: 0 = Real, 1 = Fake  (matches the flipped dataset).
    """

    def __init__(self, vit_model: ViTForImageClassification, k: int = 20):
        super().__init__()
        self.vit = vit_model.vit
        self.k = k

        n_enc = len(self.vit.encoder.layer)
        # hidden_states tuple: index 0 = patch embeddings,
        #                       index 1..n_enc = after each encoder block.
        self.probe_indices = {
            "start":  1,
            "middle": n_enc // 2,
            "end":    n_enc,
        }

        # Freeze the entire ViT
        for p in self.vit.parameters():
            p.requires_grad = False

        # Classifier: (3 layers) × k features → 2 classes
        in_dim = 3 * k
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    # ------------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W) — output of ViTImageProcessor.
        Returns:
            logits: (B, 2)
        """
        with torch.no_grad():
            outputs = self.vit(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states  # len = n_enc + 1

        lid_parts = []
        for idx in self.probe_indices.values():
            cls_token = hidden_states[idx][:, 0, :]          # (B, D)
            lid_feat = compute_lid_features(cls_token, self.k)  # (B, k)
            lid_parts.append(lid_feat)

        combined = torch.cat(lid_parts, dim=-1)               # (B, 3k)
        return self.classifier(combined)

    # ------------------------------------------------------------------
    def analyze_lid(self, pixel_values: torch.Tensor) -> dict:
        """
        Return per-layer global intrinsic dimension estimates (for logging).

        Returns:
            dict: {"start": float, "middle": float, "end": float}
        """
        with torch.no_grad():
            outputs = self.vit(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states

        results = {}
        for name, idx in self.probe_indices.items():
            cls_token = hidden_states[idx][:, 0, :]  # (B, D)
            results[name] = twonn_global_id(cls_token)
        return results

    # ------------------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.classifier.parameters() if p.requires_grad]
