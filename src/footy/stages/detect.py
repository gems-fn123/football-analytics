"""Stage 1: detect players, goalkeepers, referees, and the ball per frame.

Plain version: draw a box around every person and the ball in each still image.
No knowledge of who they are, no memory between frames.

Backend is chosen by config so the AGPL detector stays optional. See NOTICE.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from footy.schemas import DETECTIONS, validate
from footy.stages.base import Stage, StageResult


def resolve_device(requested: str) -> str:
    """Honour the configured device, but fall back rather than crash.

    configs/pipeline.yaml ships with `device: cuda`. A CPU-only torch build is a
    perfectly normal way to develop against a clip, so downgrade with a warning
    instead of dying before anything is written.
    """
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


class Detector(Stage):
    name = "detect"
    requires_gpu = True

    def setup(self) -> None:
        backend = self.cfg.get("name", "rtdetr_coco")
        self.backend = backend
        self.log.info("loading detector backend=%s licence=%s", backend, self.cfg.get("licence"))

        if backend == "rtdetr_coco":
            self._setup_rtdetr()
        elif backend == "rfdetr":
            self._setup_rfdetr()
        elif backend == "ultralytics_yolo":
            # Deliberately not implemented here. Ultralytics is AGPL-3.0, and importing it
            # into src/footy/ would relicense this package. See CONTRIBUTING.md and
            # docs/licensing.md: GPL backends are subprocess-only, wrapped under scripts/.
            raise RuntimeError(
                "the ultralytics_yolo backend is AGPL-3.0 and must not be imported into "
                "src/footy/. Wrap it in scripts/ and call it via subprocess, or use "
                "configs/detector/rtdetr_coco.yaml."
            )
        else:
            raise ValueError(f"unknown detector backend: {backend!r}")

    def _setup_rtdetr(self) -> None:
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        model_id = self.cfg.get("model_id", "PekingU/rtdetr_r18vd")
        imgsz = int(self.cfg.get("imgsz", 640))

        self.processor = RTDetrImageProcessor.from_pretrained(
            model_id, size={"height": imgsz, "width": imgsz}
        )
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).eval()
        self.torch = torch
        # COCO name -> our vocabulary. Anything unmapped is dropped.
        self.id2label = self.model.config.id2label
        self.class_map = self.cfg.get("class_map") or {"person": "player", "sports ball": "ball"}

    def _setup_rfdetr(self) -> None:
        from pathlib import Path

        weights = self.cfg.get("weights")
        if not weights or not Path(weights).exists():
            raise FileNotFoundError(
                f"detector weights not found: {weights!r}. Run scripts/download_weights.sh, "
                "or switch to configs/detector/rtdetr_coco.yaml which needs no local weights."
            )
        raise NotImplementedError(
            "rfdetr backend: load the football fine-tune here once a checkpoint exists"
        )

    def _to_device(self, device: str, half: bool) -> None:
        self.model = self.model.to(device)
        self.half = half and device.startswith("cuda")
        if self.half:
            self.model = self.model.half()
        self.device = device

    def _infer_batch(
        self, frames: list[np.ndarray], indices: list[int], threshold: float
    ) -> list[dict[str, Any]]:
        """Run one batch through the model, return rows in the DETECTIONS shape."""
        import cv2

        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        if self.half:
            inputs = inputs.to(self.torch.float16)

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        # target_sizes in (height, width) returns boxes in original frame pixels.
        target_sizes = self.torch.tensor([f.shape[:2] for f in frames]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )

        rows: list[dict[str, Any]] = []
        for frame_idx, res in zip(indices, results, strict=True):
            det_id = 0
            for score, label, box in zip(
                res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist(), strict=True
            ):
                cls = self.class_map.get(self.id2label.get(int(label), ""))
                if cls is None:
                    continue
                x1, y1, x2, y2 = box
                rows.append(
                    {
                        "frame": frame_idx,
                        "det_id": det_id,
                        "cls": cls,
                        "conf": score,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
                det_id += 1
        return rows

    def run(self, ctx: dict[str, Any]) -> StageResult:
        source = ctx["video"]
        runtime = ctx["config"].get("runtime", {}) or {}

        requested = str(runtime.get("device", "cpu"))
        device = resolve_device(requested)
        if device != requested:
            self.log.warning(
                "device=%s requested but unavailable, falling back to %s", requested, device
            )
        self._to_device(device, bool(runtime.get("half_precision", False)))

        batch_size = max(1, int(runtime.get("batch_size", 8)))
        threshold = float(self.cfg.get("conf_threshold", 0.35))
        self.log.info("device=%s batch_size=%d conf>=%.2f", self.device, batch_size, threshold)

        rows: list[dict[str, Any]] = []
        batch: list[np.ndarray] = []
        indices: list[int] = []
        processed: list[int] = []

        for frame_idx, image in source:
            batch.append(image)
            indices.append(frame_idx)
            processed.append(frame_idx)
            if len(batch) == batch_size:
                rows.extend(self._infer_batch(batch, indices, threshold))
                batch, indices = [], []
        if batch:
            rows.extend(self._infer_batch(batch, indices, threshold))

        df = validate(pd.DataFrame(rows, columns=list(DETECTIONS)), DETECTIONS, self.name)
        per_class = df["cls"].value_counts().to_dict() if len(df) else {}
        return StageResult(
            self.name,
            df,
            # The frame grid actually processed. Downstream interpolation (stages.ball)
            # needs it: the detections table alone cannot distinguish "no ball found in
            # this frame" from "this frame was never decoded".
            artifacts={"frames": processed},
            stats={
                "n_detections": len(df),
                "n_frames": len(processed),
                "per_class": per_class,
                "device": self.device,
            },
        )

    def teardown(self) -> None:
        model = getattr(self, "model", None)
        if model is not None and getattr(self, "device", "cpu").startswith("cuda"):
            self.model = model.cpu()
            self.torch.cuda.empty_cache()
