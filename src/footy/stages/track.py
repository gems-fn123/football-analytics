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

        stitch_cfg = self.cfg.get("stitch") or {}
        n_stitched = 0
        if stitch_cfg.get("enabled") and video is not None:
            df, n_stitched = self._stitch(df, video, stitch_cfg)

        tracked_rows = df[df["track_id"] >= 0]
        n_tracks = int(tracked_rows["track_id"].nunique())
        return StageResult(
            self.name,
            df,
            stats={
                "n_tracks": n_tracks,
                "n_stitched": n_stitched,
                "tracked_fraction": (
                    round(len(tracked_rows) / int(trackable.sum()), 3) if trackable.any() else 0.0
                ),
                "mean_track_len": (round(len(tracked_rows) / n_tracks, 1) if n_tracks else 0.0),
                "id_switches": None,  # needs ground truth, see docs/architecture.md
            },
        )

    def _stitch(self, df: pd.DataFrame, video: Any, cfg: dict) -> tuple[pd.DataFrame, int]:
        """Merge fragmented tracks by OSNet appearance. Off by default; see
        footy.reid.stitch for why the thresholds are conservative."""
        from pathlib import Path

        from footy.reid.embedder import Embedder
        from footy.reid.stitch import stitch_tracks

        weights = cfg.get("weights", "models/weights/osnet_x0_25.pt")
        if not Path(weights).exists():
            self.log.warning("stitch enabled but reid weights missing at %s; skipping", weights)
            return df, 0
        # Construct before the (expensive) crop decode pass, and degrade the same
        # way as missing weights: a truncated download must not kill tracking.
        try:
            embedder = Embedder(weights)
        except Exception as exc:
            self.log.warning("stitch reid weights unloadable at %s (%s); skipping", weights, exc)
            return df, 0

        tracked = df[(df["track_id"] >= 0) & (df["cls"] != "ball")]
        max_crops = int(cfg.get("max_crops", 6))
        # Pick which (frame, box) samples to crop, spread along each track.
        wanted: dict[int, list] = {}
        for _tid, group in tracked.groupby("track_id"):
            step = max(1, len(group) // max_crops)
            for row in group.iloc[::step].itertuples():
                wanted.setdefault(int(row.frame), []).append(row)

        crops: dict[int, list[np.ndarray]] = {}
        last_needed = max(wanted) if wanted else -1
        for frame_idx, image in video:
            for row in wanted.get(frame_idx, ()):
                x1, y1 = max(int(row.x1), 0), max(int(row.y1), 0)
                x2, y2 = min(int(row.x2), image.shape[1]), min(int(row.y2), image.shape[0])
                if x2 - x1 >= 8 and y2 - y1 >= 16:
                    # .copy(): a slice view would pin the whole decoded frame in
                    # memory until embedding time - gigabytes on long videos.
                    crops.setdefault(int(row.track_id), []).append(image[y1:y2, x1:x2].copy())
            if frame_idx >= last_needed:
                break

        info = {}
        for tid, group in tracked.groupby("track_id"):
            if not crops.get(int(tid)):
                continue
            emb = np.median(embedder.embed(crops[int(tid)]), axis=0)
            emb = emb / max(float(np.linalg.norm(emb)), 1e-9)
            info[int(tid)] = {
                "start": int(group["frame"].min()),
                "end": int(group["frame"].max()),
                "embedding": emb,
                "cls": group["cls"].mode().iat[0],
            }

        mapping = stitch_tracks(
            info,
            sim_threshold=float(cfg.get("sim_threshold", 0.95)),
            margin=float(cfg.get("margin", 0.08)),
            max_gap_frames=int(cfg.get("max_gap_frames", 250)),
        )
        n_merged = sum(1 for t, r in mapping.items() if t != r)
        if n_merged:
            df = df.assign(track_id=df["track_id"].map(lambda t: mapping.get(int(t), t)))
            self.log.info("stitched %d track fragments", n_merged)
        return df, n_merged
