"""Derived charts for the analyst report.

Only derived measurements are plotted - never the source frames, which are watermarked
stock previews (pre-registration C-H3). SVG so it inlines into a self-contained page.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = Path(__file__).parent / "charts"
OUT.mkdir(exist_ok=True)
CLIPS = ["clip0", "clip1", "clip2"]
GT = {"clip0": 15, "clip1": 20, "clip2": 17}
INK = "#1a1a1a"
ACCENT = {"clip0": "#2A55C4", "clip1": "#C62828", "clip2": "#0E9594"}

plt.rcParams.update(
    {
        # Explicit stack: the SVG is inlined into a page where DejaVu does not exist,
        # so name faces the browser will actually resolve rather than falling back
        # silently to something unplanned.
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "sans-serif"],
        "font.size": 9,
        "axes.edgecolor": "#999",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": "#555",
        "ytick.color": "#555",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
    }
)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.svg", format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", name)


def chart_detections() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for clip in CLIPS:
        px = pd.read_parquet(PROCESSED / clip / "tracks_px.parquet")
        p = px[px["cls"] != "ball"]
        per_frame = p.groupby("frame").size()
        t = per_frame.index / 25.0
        ax.plot(t, per_frame.rolling(25, min_periods=1).mean(), lw=1.6,
                color=ACCENT[clip], label=f"{clip} (GT {GT[clip]})")
        ax.axhline(GT[clip], color=ACCENT[clip], ls=":", lw=0.9, alpha=0.55)
    ax.axhline(10, color="#B00020", ls="--", lw=1.1)
    ax.text(0.3, 10.4, "C-B1 floor = 10", color="#B00020", fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("persons detected\n(1 s rolling mean)")
    ax.set_ylim(0, 28)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower right")
    save(fig, "detections")


def chart_calibration() -> None:
    blob = json.loads((Path(__file__).parent / "calibration_feasibility.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for clip in CLIPS:
        s = pd.DataFrame(blob[clip]["series"])
        ax.plot(s["t_s"], s["median_err_m"], lw=1.6, color=ACCENT[clip], label=clip)
        b = blob[clip]["first_breach_of_2m_s"]
        if b is not None:
            ax.plot([b], [2.0], "o", ms=5, color=ACCENT[clip])
    ax.axhline(2.0, color="#B00020", ls="--", lw=1.3)
    ax.text(0.4, 2.4, "C-F1 limit = 2.0 m", color="#B00020", fontsize=8)
    ax.set_yscale("symlog", linthresh=2)
    ax.set_xlabel("time since reference frame (s)")
    ax.set_ylabel("position error of a static\nhomography (m, median)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    save(fig, "calibration_drift")


def chart_track_life() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, clip in zip(axes, CLIPS):
        px = pd.read_parquet(PROCESSED / clip / "tracks_px.parquet")
        p = px[(px["cls"] != "ball") & (px["track_id"] >= 0)]
        life = p.groupby("track_id")["frame"].agg(lambda s: s.max() - s.min() + 1) / 25.0
        ax.hist(life, bins=np.linspace(0, life.max(), 24), color=ACCENT[clip], alpha=0.85)
        ax.axvline(1.0, color="#B00020", ls="--", lw=1.0)
        ax.set_title(f"{clip}  n={len(life)}", fontsize=9)
        ax.set_xlabel("track lifetime (s)")
    axes[0].set_ylabel("tracks")
    save(fig, "track_life")


def chart_occupancy() -> None:
    """Where each side spent time, in image space. Not metres - deliberately."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.9))
    for ax, clip in zip(axes, CLIPS):
        px = pd.read_parquet(PROCESSED / clip / "tracks_px.parquet")
        p = px[(px["cls"] != "ball") & px["team"].isin(["home", "away"])]
        for team, colr in (("home", "#C62828"), ("away", "#1E64C8")):
            q = p[p["team"] == team]
            cx = (q["x1"] + q["x2"]) / 2
            cy = q["y2"]
            ax.scatter(cx, cy, s=1.5, alpha=0.06, color=colr, edgecolors="none")
        ax.set_xlim(0, 768)
        ax.set_ylim(432, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(clip, fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#bbb")
    save(fig, "occupancy")


if __name__ == "__main__":
    chart_detections()
    chart_calibration()
    chart_track_life()
    # chart_occupancy() is deliberately not shipped: in image space, with a panning
    # camera, a position density is not a heat map - it is mostly a picture of where
    # the camera pointed. Publishing it would invite exactly the misreading this
    # report exists to prevent. Kept in source for anyone who wants to see why.
    print("charts ->", OUT)
