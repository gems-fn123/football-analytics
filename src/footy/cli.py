"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from footy.config import Config
from footy.io.video import probe
from footy.logging_utils import get_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="footy", description="Football video to analytics")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full pipeline on a video")
    run.add_argument("--video", required=True)
    run.add_argument("--config", default="configs/pipeline.yaml")
    run.add_argument("--match", default=None, help="match config with the squad list")
    run.add_argument("--max-frames", type=int, default=None, help="smoke test budget")

    inspect = sub.add_parser("inspect", help="probe a video without processing it")
    inspect.add_argument("--video", required=True)

    calib = sub.add_parser("calibrate", help="solve a static homography for a fixed camera")
    calib.add_argument("--video", required=True)
    calib.add_argument("--out", default="configs/camera/fixed_wide_points.json")

    xg = sub.add_parser("fit-xg", help="recalibrate the xG model on local shots")
    xg.add_argument("--shots", required=True, help="parquet or csv of local shots")
    xg.add_argument("--out", default="models/weights/xg_local.pkl")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = get_logger()

    if args.command == "inspect":
        meta = probe(args.video)
        print(f"{meta.path.name}: {meta.width}x{meta.height} @ {meta.fps:.2f} fps, "
              f"{meta.n_frames} frames, {meta.duration_s:.1f} s")
        return 0

    if args.command == "run":
        from footy.pipeline import run_pipeline

        cfg = Config.load(args.config, args.match)
        if args.max_frames is not None:
            cfg.raw.setdefault("io", {})["max_frames"] = args.max_frames
        result = run_pipeline(args.video, cfg)
        log.info("wrote %s", result["output_dir"])
        return 0

    if args.command == "calibrate":
        log.error("not implemented, see scripts/calibrate_fixed_camera.py")
        return 1

    if args.command == "fit-xg":
        from footy.analytics.xg import fit_local

        fit_local(args.shots, args.out)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
