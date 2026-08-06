"""Compute the match analytics and render the map-view charts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pitch_model as pm  # noqa: E402
from metrics import (  # noqa: E402
    FPS,
    add_kinematics,
    assign_synthetic_identities,
    noise_floor,
    player_table,
    possession,
    stitch_tracks,
    team_shape,
)

ROOT = HERE.parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
CHARTS = OUT / "charts"
CHARTS.mkdir(exist_ok=True)

HOME_C = "#1F4FA8"   # clip0 home kit is blue
AWAY_C = "#C8CDD4"   # away kit is white
INK = "#101820"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "sans-serif"],
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#555", "ytick.color": "#555",
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
})


def draw_pitch(ax, lw=1.0, colour="#7A8894"):
    for _, poly in pm.polylines():
        ax.plot(poly[:, 0], poly[:, 1], color=colour, lw=lw, zorder=1, solid_capstyle="round")
    ax.set_xlim(-4, pm.LENGTH + 4)
    ax.set_ylim(-4, pm.WIDTH + 4)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig, name):
    fig.savefig(CHARTS / f"{name}.svg", format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("  chart:", name)


def main() -> int:
    clip = sys.argv[1] if len(sys.argv) > 1 else "clip0"
    df = pd.read_parquet(ROOT / "data/processed" / clip / "tracks_pitch.parquet")

    people = df[(df["cls"] != "ball") & (df["track_id"] >= 0)].copy()
    people = people[pm.in_bounds(people["x_m"].to_numpy(), people["y_m"].to_numpy(), 3.0)]
    life = people.groupby("track_id")["t_s"].agg(lambda s: s.max() - s.min())
    people = people[people["track_id"].isin(life[life >= 2.0].index)]

    k = add_kinematics(people)
    nf = noise_floor(k)
    n_raw_tracks = int(k["track_id"].nunique())
    k = stitch_tracks(k)
    n_stitched = int(k["player_id"].nunique())
    k = assign_synthetic_identities(k)
    print(f"  tracks: {n_raw_tracks} raw -> {n_stitched} after stitching -> "
          f"{k['synth_id'].nunique()} named")

    ball = df[df["cls"] == "ball"].copy()
    ball = ball[pm.in_bounds(ball["x_m"].to_numpy(), ball["y_m"].to_numpy(), 3.0)]
    ball = ball.sort_values("frame").groupby("frame", as_index=False).first()

    pt = player_table(k)
    shape = team_shape(k)
    poss = possession(k, ball)

    teams = [t for t in ("home", "away") if (k["team"] == t).any()]
    summary = {
        "clip": clip,
        "duration_s": round(float(k["t_s"].max() - k["t_s"].min()), 1),
        "frames": int(k["frame"].nunique()),
        "tracked_players": int(k["synth_id"].nunique()),
        "noise_floor": nf,
        "mean_players_on_camera": round(float(k.groupby("frame").size().mean()), 1),
    }

    # -------- possession and territory
    if len(poss):
        share = poss["team"].value_counts(normalize=True).to_dict()
        summary["possession_pct"] = {t: round(100 * share.get(t, 0.0), 1) for t in teams}
        summary["possession_frames"] = int(len(poss))
    else:
        summary["possession_pct"] = {}

    # Thirds are named from each side's own point of view. pm.thirds() assumes a team
    # attacking left-to-right, so the side attacking the other way has its x mirrored -
    # otherwise one team's "attacking third" is reported as its defensive third.
    cent = {t: float(k.loc[k["team"] == t, "x_s"].mean()) for t in teams}
    attack_right = {t: (t == min(cent, key=cent.get)) for t in teams} if len(cent) == 2 \
        else {t: cent[t] < pm.CX for t in teams}
    summary["attacking_direction"] = {
        t: ("left_to_right" if attack_right[t] else "right_to_left") for t in teams
    }

    terr = {}
    for t in teams:
        d = k[k["team"] == t]
        x = d["x_s"].to_numpy()
        th = pm.thirds(x if attack_right[t] else (pm.LENGTH - x))
        terr[t] = {z: round(100 * float((th == z).mean()), 1)
                   for z in ("defensive", "middle", "attacking")}
    summary["territory_pct"] = terr

    # -------- team shape
    sh = {}
    for t in teams:
        s = shape[shape["team"] == t]
        if len(s):
            sh[t] = {
                "mean_width_m": round(float(s["width_m"].mean()), 1),
                "mean_depth_m": round(float(s["depth_m"].mean()), 1),
                "mean_centroid_x_m": round(float(s["cx"].mean()), 1),
                "mean_centroid_y_m": round(float(s["cy"].mean()), 1),
                "mean_players_visible": round(float(s["n"].mean()), 1),
            }
    summary["shape"] = sh

    # -------- physical
    if len(pt):
        summary["physical"] = {
            "median_m_per_min": round(float(pt["m_per_min"].median()), 1),
            "max_top_speed_ms": round(float(pt["top_speed_ms"].max()), 2),
            "total_sprints": int(pt["sprints"].sum()),
            "players_reported": int(len(pt)),
        }

    # ---------------------------------------------------------------- charts
    # 1. average positions / formation
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1))
    for ax, t in zip(axes, teams):
        draw_pitch(ax)
        d = pt[pt["team"] == t]
        col = HOME_C if t == "home" else AWAY_C
        ax.scatter(d["mean_x_m"], d["mean_y_m"], s=280, color=col,
                   edgecolors=INK, linewidths=0.8, zorder=3)
        for _, r in d.iterrows():
            ax.text(r["mean_x_m"], r["mean_y_m"], r["player"][1:], ha="center", va="center",
                    fontsize=7.5, zorder=4,
                    color="white" if t == "home" else INK, fontweight="bold")
        ax.set_title(f"{t} — average positions", fontsize=10)
    save(fig, "formation")

    # 2. heat maps, now legitimate because positions are in pitch metres
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1))
    for ax, t in zip(axes, teams):
        draw_pitch(ax, colour="#9AA6B2")
        d = k[k["team"] == t]
        ax.hist2d(d["x_s"], d["y_s"], bins=[np.linspace(0, pm.LENGTH, 26),
                                            np.linspace(0, pm.WIDTH, 17)],
                  cmap="YlOrRd" if t == "home" else "PuBu", alpha=0.85, zorder=0)
        ax.set_title(f"{t} — occupancy (pitch metres)", fontsize=10)
    save(fig, "heatmap")

    # 3. ball trajectory in map view
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    draw_pitch(ax)
    if len(ball):
        ax.plot(ball["x_m"], ball["y_m"], color="#B5731A", lw=1.0, alpha=0.55, zorder=2)
        sc = ax.scatter(ball["x_m"], ball["y_m"], c=ball["t_s"], cmap="inferno",
                        s=11, zorder=3)
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("time (s)", fontsize=8)
        cb.outline.set_visible(False)
    ax.set_title("Ball path, normalised to the pitch", fontsize=10)
    save(fig, "ball_path")

    # 4. team centroid over time
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    for t in teams:
        s = shape[shape["team"] == t].sort_values("t_s")
        ax.plot(s["t_s"], s["cx"], lw=1.5,
                color=HOME_C if t == "home" else "#7E8894", label=t)
    ax.axhline(pm.CX, color="#B00020", ls="--", lw=0.9)
    ax.text(0.2, pm.CX + 1.5, "halfway", color="#B00020", fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("team centroid\n(m from left goal)")
    ax.set_ylim(0, pm.LENGTH)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "centroid")

    # 5. physical: distance vs top speed
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for t in teams:
        d = pt[pt["team"] == t]
        ax.scatter(d["m_per_min"], d["top_speed_ms"], s=70,
                   color=HOME_C if t == "home" else "#9AA6B2",
                   edgecolors=INK, linewidths=0.7, label=t, zorder=3)
        for _, r in d.iterrows():
            ax.annotate(r["player"], (r["m_per_min"], r["top_speed_ms"]),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("metres per minute (active play)")
    ax.set_ylabel("top speed (m/s)")
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "physical")

    # ---------------------------------------------------------------- write
    pt.to_csv(OUT / f"{clip}_players.csv", index=False)
    (OUT / f"{clip}_summary.json").write_text(json.dumps(summary, indent=2))
    k.to_parquet(OUT / f"{clip}_tracks_final.parquet", index=False)
    if len(poss):
        poss.to_csv(OUT / f"{clip}_possession.csv", index=False)

    print(json.dumps(summary, indent=2))
    print("\nplayers:", len(pt))
    print(pt.head(24).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
