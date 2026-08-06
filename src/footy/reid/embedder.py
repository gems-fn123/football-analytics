"""Crop -> appearance embedding via OSNet.

ImageNet-initialised weights work unmodified for coarse same-or-different-player
judgements; a soccer-specific fine-tune on SoccerNet reid-2023 slots in as a
weights swap. Weights download: scripts/download_weights.sh.
"""

from __future__ import annotations

import numpy as np

# Standard person-reid input geometry and ImageNet statistics.
INPUT_HW = (256, 128)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Embedder:
    def __init__(self, weights: str | None = None, device: str = "cpu") -> None:
        import torch

        from footy.reid.osnet import osnet_x0_25

        self.model = osnet_x0_25(num_classes=1, pretrained=False, loss="triplet")
        if weights:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
            # Classifier head sizes differ per training run and are unused for
            # embeddings; keep everything that matches.
            own = self.model.state_dict()
            usable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
            self.model.load_state_dict(usable, strict=False)
            self.loaded_fraction = len(usable) / len(own)
        else:
            self.loaded_fraction = 0.0
        self.model.eval().to(device)
        self.device = device
        self.torch = torch

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """L2-normalised embeddings, one row per crop."""
        import cv2

        batch = []
        for crop in crops_bgr:
            resized = cv2.resize(crop, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
        with self.torch.no_grad():
            x = self.torch.from_numpy(np.stack(batch)).to(self.device)
            feats = self.model(x).cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.clip(norms, 1e-9, None)
