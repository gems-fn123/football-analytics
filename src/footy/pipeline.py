"""Run-time orchestrator.

Reads a video and a match config. Writes tracks, events, and a report. Touches no
provider API at any point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from footy.config import Config
from footy.io.video import VideoSource
from footy.io.writers import write_table
from footy.logging_utils import get_logger
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

    if outputs.get("tracks", True) and "calibrate" in ctx:
        written["tracks"] = write_table(ctx["calibrate"].table, out_dir / "tracks.parquet")
    if outputs.get("events", True) and "events" in ctx:
        written["events"] = write_table(ctx["events"].table, out_dir / "events.parquet")

    return {"context": ctx, "outputs": written, "output_dir": out_dir}
