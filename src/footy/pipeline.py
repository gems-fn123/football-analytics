"""Run-time orchestrator.

Reads a video and a match config. Writes tracks, events, and a report. Touches no
provider API at any point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from footy.config import Config
from footy.io.video import VideoSource
from footy.io.writers import write_table
from footy.logging_utils import get_logger
from footy.schemas import TRACKS_M, validate
from footy.stages.ball import BallTracker
from footy.stages.calibrate import Calibrator
from footy.stages.detect import Detector
from footy.stages.events import EventBuilder
from footy.stages.identity import IdentityResolver
from footy.stages.team import TeamAssigner
from footy.stages.track import Tracker

STAGE_ORDER = [
    ("detect", Detector, "detector"),
    ("track", Tracker, "tracker"),
    ("team", TeamAssigner, "team"),
    ("identity", IdentityResolver, "identity"),
    ("calibrate", Calibrator, "camera"),
    ("ball", BallTracker, "ball"),
    ("events", EventBuilder, "events"),
]

# For tracks_px.parquet: the richest pixel-space table that actually ran.
PX_TABLE_PREFERENCE = ("identity", "team", "track", "detect")


def _final_tracks_m(ctx: dict[str, Any]) -> pd.DataFrame:
    """Calibrated table -> smoothed positions and speed, in contract shape."""
    from footy.analytics.physical import add_kinematics

    df = add_kinematics(ctx["calibrate"].table)
    return validate(df[list(TRACKS_M)], TRACKS_M, "tracks_m")


def _report_tables(ctx: dict[str, Any], tracks_m: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    stage_rows = [
        {"stage": key, "rows": len(ctx[key].table), "stats": str(ctx[key].stats)}
        for key, _, _ in STAGE_ORDER
        if key in ctx
    ]
    tables["Run summary"] = pd.DataFrame(stage_rows)

    for key in PX_TABLE_PREFERENCE:
        if key in ctx and "team" in ctx[key].table.columns:
            px = ctx[key].table
            person = px[px["cls"] != "ball"]
            split = (
                person.groupby("team", dropna=False)
                .agg(rows=("frame", "size"), tracks=("track_id", "nunique"))
                .reset_index()
            )
            split["team"] = split["team"].fillna("(unassigned)")
            tables["Team assignment"] = split
            break

    if tracks_m is not None and tracks_m["x_m"].notna().any():
        from footy.analytics.physical import summarise_player

        tables["Physical per player"] = summarise_player(tracks_m)

    if "ball" in ctx and len(ctx["ball"].table):
        tables["Ball trajectory (head)"] = ctx["ball"].table.head(25)

    return tables


def run_pipeline(video: str | Path, cfg: Config) -> dict[str, Any]:
    log = get_logger("footy.pipeline")
    io_cfg = cfg.get("io", {})
    enabled = cfg.get("stages", {})

    source = VideoSource(
        video,
        stride=io_cfg.get("frame_stride", 1),
        max_frames=io_cfg.get("max_frames"),
    )
    log.info(
        "ingest %s  %dx%d  %.2f fps  %.1f s",
        source.meta.path.name,
        source.meta.width,
        source.meta.height,
        source.meta.fps,
        source.meta.duration_s,
    )

    ctx: dict[str, Any] = {"video": source, "config": cfg}

    for key, cls, cfg_key in STAGE_ORDER:
        if not enabled.get(key, True):
            log.info("skip stage=%s", key)
            continue
        stage = cls(cfg.get(cfg_key, {}) or {})
        stage.setup()
        try:
            result = stage.run(ctx)
        finally:
            stage.teardown()
        ctx[key] = result
        log.info("stage=%s rows=%d stats=%s", key, len(result.table), result.stats)

    out_dir = Path(io_cfg.get("output_dir", "data/processed")) / cfg.match_id
    outputs = cfg.get("outputs", {})
    written: dict[str, Path] = {}

    tracks_m: pd.DataFrame | None = None
    if outputs.get("tracks", True):
        # Pixel-space truth is always worth keeping; it is all there is when the
        # camera is uncalibrated, and it is the debugging ground truth when not.
        for key in PX_TABLE_PREFERENCE:
            if key in ctx:
                written["tracks_px"] = write_table(ctx[key].table, out_dir / "tracks_px.parquet")
                break
        if "calibrate" in ctx:
            tracks_m = _final_tracks_m(ctx)
            written["tracks"] = write_table(tracks_m, out_dir / "tracks.parquet")

    if outputs.get("ball", True) and "ball" in ctx:
        written["ball"] = write_table(ctx["ball"].table, out_dir / "ball.parquet")

    if outputs.get("events", True) and "events" in ctx:
        written["events"] = write_table(ctx["events"].table, out_dir / "events.parquet")

    if outputs.get("report", True):
        from footy.viz.report import build_report

        identity_pct = 0.0
        if "identity" in ctx:
            identity_pct = float(ctx["identity"].stats.get("resolved_fraction", 0.0))
        written["report"] = build_report(
            cfg.match_id,
            _report_tables(ctx, tracks_m),
            out_dir / "report.html",
            identity_pct=identity_pct,
        )

    return {"context": ctx, "outputs": written, "output_dir": out_dir}
