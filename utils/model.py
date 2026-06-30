"""Multi-layer LID deepfake classifier.

Frozen ViT CLS tokens at 3 depths -> per-point LID measured against a fixed
real-image reference bank -> shallow MLP. Only the MLP is trained. See README.
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification

from .lid_estimator import compute_lid_features, twonn_global_id


class MultiLayerLIDModel(nn.Module):
    def __init__(self, vit_model: ViTForImageClassification, k: int = 20):
        super().__init__()
        self.vit = vit_model.vit
        self.k = k

        n_enc = len(self.vit.encoder.layer)
        # hidden_states[0] = embeddings; [1..n_enc] = after each encoder block.
        self.probe_indices = {"start": 1, "middle": n_enc // 2, "end": n_enc}

        for p in self.vit.parameters():
            p.requires_grad = False

        self._has_reference = False
        self.classifier = nn.Sequential(
            nn.Linear(3 * k, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2),
        )

    def _extract_cls(self, pixel_values: torch.Tensor) -> dict:
        with torch.no_grad():
            hs = self.vit(pixel_values=pixel_values,
                          output_hidden_states=True).hidden_states
        return {name: hs[idx][:, 0, :] for name, idx in self.probe_indices.items()}

    @torch.no_grad()
    def build_reference_bank(self, ref_loader, device: str, max_size: int = 2048) -> dict:
        # Freeze CLS features of a fixed real-image pool as the LID reference.
        was_training = self.training
        self.eval()
        banks = {name: [] for name in self.probe_indices}
        count = 0
        for inputs, _ in ref_loader:
            feats = self._extract_cls(inputs["pixel_values"].to(device))
            for name in self.probe_indices:
                banks[name].append(feats[name].cpu())
            count += inputs["pixel_values"].size(0)
            if count >= max_size:
                break
        banks = {name: torch.cat(v)[:max_size] for name, v in banks.items()}
        self.load_reference_bank(banks)
        if was_training:
            self.train()
        return {name: tuple(v.shape) for name, v in banks.items()}

    def load_reference_bank(self, banks: dict) -> None:
        for name in self.probe_indices:
            feats = banks[name].detach().clone()
            buf = f"ref_{name}"
            if hasattr(self, buf):
                getattr(self, buf).resize_(feats.shape).copy_(feats)
            else:
                self.register_buffer(buf, feats, persistent=True)
        self._has_reference = True

    def reference_bank(self) -> dict:
        if not self._has_reference:
            raise RuntimeError("Reference bank not built/loaded.")
        return {name: getattr(self, f"ref_{name}").clone() for name in self.probe_indices}

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not self._has_reference:
            raise RuntimeError("Install a reference bank before forward().")
        feats = self._extract_cls(pixel_values)
        parts = [
            compute_lid_features(feats[name],
                                 getattr(self, f"ref_{name}").to(feats[name].device),
                                 self.k)
            for name in self.probe_indices
        ]
        return self.classifier(torch.cat(parts, dim=-1))

    def analyze_lid(self, pixel_values: torch.Tensor) -> dict:
        feats = self._extract_cls(pixel_values)
        return {name: twonn_global_id(feats[name]) for name in self.probe_indices}

    def trainable_parameters(self):
        return [p for p in self.classifier.parameters() if p.requires_grad]
