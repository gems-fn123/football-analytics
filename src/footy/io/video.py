"""Stage 0: ingest. Decode the recording and hand out frames.

This is the stage the first schema diagram was missing. It owns everything that is
about the *file* rather than about football: container, codec, fps, resolution,
stride, and (for broadcast) where the cuts and replays are.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: Path
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoMeta:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    meta = VideoMeta(
        path=path,
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


class VideoSource:
    """Frame iterator with stride and frame budget."""

    def __init__(
        self,
        path: str | Path,
        stride: int = 1,
        max_frames: int | None = None,
        start_frame: int = 0,
    ) -> None:
        self.meta = probe(path)
        self.stride = max(1, stride)
        self.max_frames = max_frames
        self.start_frame = start_frame

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        cap = cv2.VideoCapture(str(self.meta.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        idx, emitted = self.start_frame, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if (idx - self.start_frame) % self.stride == 0:
                    yield idx, frame
                    emitted += 1
                    if self.max_frames and emitted >= self.max_frames:
                        break
                idx += 1
        finally:
            cap.release()


def detect_shot_boundaries(path: str | Path, threshold: float = 0.55) -> list[int]:
    """Broadcast only: find hard cuts via colour-histogram distance.

    TODO: replaces nothing yet. Returned indices are candidate cut points; the
    caller decides whether a segment is live play or a replay. Replay detection
    proper needs a logo-wipe or slow-motion classifier.
    """
    raise NotImplementedError("shot boundary detection not implemented")
