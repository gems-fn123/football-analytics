"""Stage 2: link detections across frames into tracks.

Plain version: keep the same number on the same player from one frame to the next.
Pure geometry and appearance, no external data.

Known failure: identity switches when players cross or are hidden. In the box on a
corner, expect swaps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from footy.schemas import TRACKS_PX, validate
from footy.stages.base import Stage, StageResult, upstream_table

# The ball is deliberately excluded: a 10-pixel object that teleports between frames
# breaks motion-model association. stages.ball owns the ball trajectory instead.
DEFAULT_TRACK_CLASSES = ("player", "goalkeeper", "referee")

# sv.Detections wants integer class ids. Stable order, not semantic.
CLASS_IDS = {"player": 0, "goalkeeper": 1, "referee": 2, "ball": 3}


class Tracker(Stage):
    name = "track"

    def setup(self) -> None:
        backend = self.cfg.get("name", "bytetrack")
        self.backend = backend
        self.log.info("tracker=%s licence=%s", backend, self.cfg.get("licence"))

        if backend == "botsort":
            raise NotImplementedError(
                "botsort needs a re-ID backbone that supervision does not ship. Use "
                "configs/tracker/bytetrack.yaml, or wrap BoT-SORT under scripts/."
            )
        if backend != "bytetrack":
            raise ValueError(f"unknown tracker backend: {backend!r}")

        import supervision as sv

        self.track_classes = set(self.cfg.get("track_classes") or DEFAULT_TRACK_CLASSES)
        self.sv = sv

    def _make_tracker(self, effective_fps: float):
        """Built per run: frame_rate must reflect the video and the decode stride,
        or lost_track_buffer ages in the wrong time units (it is specified in frames
        at the effective rate; supervision converts it to wall-clock internally)."""
        return self.sv.ByteTrack(
            track_activation_threshold=float(self.cfg.get("track_activation_threshold", 0.25)),
            lost_track_buffer=int(self.cfg.get("lost_track_buffer", 60)),
            minimum_matching_threshold=float(self.cfg.get("minimum_matching_threshold", 0.85)),
            frame_rate=max(1, round(effective_fps)),
        )

    def _track_frame(self, group: pd.DataFrame) -> dict[int, int]:
        """Feed one frame to the tracker, return det_id -> track_id for that frame."""
        if group.empty:
            # Still update, so lost_track_buffer ages in frames rather than in
            # frames-that-happened-to-have-detections.
            self.tracker.update_with_detections(self.sv.Detections.empty())
            return {}

        dets = self.sv.Detections(
            xyxy=group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32),
            confidence=group["conf"].to_numpy(dtype=np.float32),
            class_id=group["cls"].map(CLASS_IDS).fillna(0).to_numpy(dtype=int),
            data={"det_id": group["det_id"].to_numpy()},
        )
        tracked = self.tracker.update_with_detections(dets)
        if tracked.tracker_id is None or len(tracked) == 0:
            return {}
        return {
            int(det_id): int(track_id)
            for det_id, track_id in zip(tracked.data["det_id"], tracked.tracker_id, strict=True)
        }

    def run(self, ctx: dict[str, Any]) -> StageResult:
        detections = upstream_table(ctx, "detect", stage=self.name)

        video = ctx.get("video")
        fps = float(getattr(video.meta, "fps", 0) or 25.0) if video is not None else 25.0
        stride = max(1, int(getattr(video, "stride", 1))) if video is not None else 1
        self.tracker = self._make_tracker(fps / stride)

        # -1 is the contract's "untracked" value; ball rows keep it by design.
        track_ids = pd.Series(-1, index=detections.index, dtype="int64")
        trackable = detections["cls"].isin(self.track_classes)

        # Walk the full processed-frame grid, not just frames that had detections:
        # a frame where the detector found nothing must still age the tracker, or a
        # long dropout leaves stale tracks alive to steal the next player's identity.
        grid = ctx["detect"].artifacts.get("frames") or sorted(detections["frame"].unique())
        by_frame = dict(iter(detections[trackable].groupby("frame")))
        empty = detections.iloc[0:0]

        for frame_idx in grid:
            group = by_frame.get(frame_idx, empty)
            mapping = self._track_frame(group)
            if mapping:
                track_ids.loc[group.index] = group["det_id"].map(mapping).fillna(-1).astype("int64")

        df = validate(detections.assign(track_id=track_ids), TRACKS_PX, self.name)

        tracked_rows = df[df["track_id"] >= 0]
        n_tracks = int(tracked_rows["track_id"].nunique())
        return StageResult(
            self.name,
            df,
            stats={
                "n_tracks": n_tracks,
                "tracked_fraction": (
                    round(len(tracked_rows) / int(trackable.sum()), 3) if trackable.any() else 0.0
                ),
                "mean_track_len": (round(len(tracked_rows) / n_tracks, 1) if n_tracks else 0.0),
                "id_switches": None,  # needs ground truth, see docs/architecture.md
            },
        )
