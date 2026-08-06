"""Verify C-D3 numerically and characterise the criteria that failed.

C-D3 (home/away mapping correctness) was pre-registered as a qualitative eye check.
Eyeballing is weak, so it is done numerically instead: recompute the torso chroma of
every assigned track, average per assigned label, and check that the 'home' group
really does sit nearer the configured home kit hex than the away hex. If the mapping
were swapped, this test would say so.

Also quantifies the two failure modes (track fragmentation, ball/team coverage) and
summarises the calibration drift series.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from footy.stages.team import hex_to_ab, torso_chroma  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
VIDEOS = {
    "clip0": ROOT / "data/raw/smoke_clip.mp4",
    "clip1": ROOT / "data/raw/smoke_clip (1).mp4",
    "clip2": ROOT / "data/raw/smoke_clip (2).mp4",
}
SAMPLE_STRIDE = 15


def verify_mapping(clip: str) -> dict:
    match = yaml.safe_load((ROOT / f"configs/match/{clip}.yaml").read_text())
    px = pd.read_parquet(PROCESSED / clip / "tracks_px.parquet")
    px = px[(px["cls"] != "ball") & (px["track_id"] >= 0) & px["team"].notna()]
    wanted = px[px["frame"] % SAMPLE_STRIDE == 0]
    by_frame = {int(f): g for f, g in wanted.groupby("frame")}

    cap = cv2.VideoCapture(str(VIDEOS[clip]))
    samples: dict[str, list[np.ndarray]] = {}
    for fidx in sorted(by_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, img = cap.read()
        if not ok:
            continue
        for row in by_frame[fidx].itertuples():
            ab = torso_chroma(img, (row.x1, row.y1, row.x2, row.y2))
            if ab is not None:
                samples.setdefault(str(row.team), []).append(ab)
    cap.release()

    observed = {k: np.median(np.stack(v), axis=0) for k, v in samples.items() if v}
    home_hex = (match.get("home") or {}).get("kit_colour")
    away_hex = (match.get("away") or {}).get("kit_colour")
    out: dict = {
        "clip": clip,
        "home_name": (match.get("home") or {}).get("name"),
        "away_name": (match.get("away") or {}).get("name"),
        "home_hex": home_hex,
        "away_hex": away_hex,
        "n_crops": {k: len(v) for k, v in samples.items()},
        "observed_chroma": {k: [round(float(x), 2) for x in v] for k, v in observed.items()},
    }
    if home_hex and away_hex and "home" in observed and "away" in observed:
        h_ref, a_ref = hex_to_ab(home_hex), hex_to_ab(away_hex)

        # Naive absolute-distance test, kept only to show why it must not be used.
        # Video colours come out muted, so a muted blue sits nearer neutral grey than
        # near saturated blue. Against a white (neutral) reference this test declares
        # a correct assignment "swapped" - the exact failure mode team.py's docstring
        # warns about. Recorded, but not the verdict.
        naive = (
            float(np.linalg.norm(observed["home"] - h_ref))
            < float(np.linalg.norm(observed["home"] - a_ref))
        ) and (
            float(np.linalg.norm(observed["away"] - a_ref))
            < float(np.linalg.norm(observed["away"] - h_ref))
        )

        # Sound test: project each observed group onto the away->home reference axis.
        # Muting shrinks the magnitude of a colour's displacement from neutral but
        # preserves its direction, so a projection is invariant to it. The home group
        # must sit further along the axis toward the home kit than the away group.
        axis = h_ref - a_ref
        denom = float(axis @ axis)
        t_home = float((observed["home"] - a_ref) @ axis / denom)
        t_away = float((observed["away"] - a_ref) @ axis / denom)
        correct = t_home > t_away
        out.update(
            {
                "naive_absolute_distance_says_correct": bool(naive),
                "axis_projection_t_home": round(t_home, 3),
                "axis_projection_t_away": round(t_away, 3),
                "axis_separation": round(t_home - t_away, 3),
                "C-D3_mapping_correct": bool(correct),
            }
        )
    return out


def characterise_tracks(clip: str) -> dict:
    px = pd.read_parquet(PROCESSED / clip / "tracks_px.parquet")
    p = px[(px["cls"] != "ball") & (px["track_id"] >= 0)]
    life = p.groupby("track_id")["frame"].agg(lambda s: s.max() - s.min() + 1)
    n_frames = px["frame"].nunique()
    return {
        "clip": clip,
        "n_tracks": int(len(life)),
        "median_life_frames": float(life.median()),
        "life_p10": float(life.quantile(0.10)),
        "life_p90": float(life.quantile(0.90)),
        "tracks_under_1s": int((life < 25).sum()),
        "tracks_under_1s_pct": round(float((life < 25).mean()), 3),
        "tracks_over_5s": int((life > 125).sum()),
        "longest_life_frames": int(life.max()),
        "clip_frames": int(n_frames),
        # A useful stability read: what share of all person-rows belong to tracks
        # that survive at least 2 s? Those are the ones worth analysing.
        "rows_in_tracks_over_2s_pct": round(
            float(p[p["track_id"].isin(life[life >= 50].index)].shape[0] / len(p)), 3
        ),
    }


def summarise_calibration() -> dict:
    blob = json.loads((Path(__file__).parent / "calibration_feasibility.json").read_text())
    out = {}
    for clip, r in blob.items():
        if "error" in r:
            out[clip] = r
            continue
        s = pd.DataFrame(r["series"])
        # The max over grid points explodes near the vanishing point, where a
        # homography is numerically unstable and the "error" is not physical.
        # The median over pitch grid points is the honest statistic.
        out[clip] = {
            "median_err_m_overall": r["median_err_m_overall"],
            "first_breach_of_2m_s": r["first_breach_of_2m_s"],
            "fraction_frames_over_2m": r["fraction_frames_over_2m"],
            "err_m_at_1s": round(float(s.loc[(s["t_s"] - 1.0).abs().idxmin(), "median_err_m"]), 2),
            "err_m_at_5s": round(float(s.loc[(s["t_s"] - 5.0).abs().idxmin(), "median_err_m"]), 2),
            "err_m_at_10s": round(float(s.loc[(s["t_s"] - 10.0).abs().idxmin(), "median_err_m"]), 2),
            "err_m_final": r["final_median_err_m"],
            "m_per_px_at_median_depth": r["scale"]["m_per_px_at_median_y"],
            "box_h_px_median": r["scale"]["box_h_px_median"],
            "verdict": r["verdict_C_F1"],
        }
    return out


def main() -> int:
    result = {"mapping": {}, "tracks": {}, "calibration": summarise_calibration()}
    for clip in VIDEOS:
        result["mapping"][clip] = verify_mapping(clip)
        result["tracks"][clip] = characterise_tracks(clip)

    print("=== C-D3 team mapping verification (numerical) ===")
    for clip, m in result["mapping"].items():
        if "C-D3_mapping_correct" in m:
            print(
                f"{clip}: {m['home_name']}(home) vs {m['away_name']}(away)  "
                f"t_home={m['axis_projection_t_home']} t_away={m['axis_projection_t_away']} "
                f"(sep {m['axis_separation']})  => "
                f"{'CORRECT' if m['C-D3_mapping_correct'] else 'SWAPPED'}"
                f"   [naive abs-distance test would say "
                f"{'correct' if m['naive_absolute_distance_says_correct'] else 'SWAPPED - unreliable'}]"
            )
        else:
            print(f"{clip}: insufficient data -> {m.get('n_crops')}")

    print("\n=== track fragmentation ===")
    for clip, t in result["tracks"].items():
        print(
            f"{clip}: {t['n_tracks']} ids over {t['clip_frames']}f | median life {t['median_life_frames']}f | "
            f"<1s: {t['tracks_under_1s']} ({t['tracks_under_1s_pct']*100:.0f}%) | "
            f">5s: {t['tracks_over_5s']} | longest {t['longest_life_frames']}f | "
            f"rows in >=2s tracks: {t['rows_in_tracks_over_2s_pct']*100:.0f}%"
        )

    print("\n=== calibration drift (median over pitch grid) ===")
    for clip, c in result["calibration"].items():
        print(
            f"{clip}: 1s {c['err_m_at_1s']}m | 5s {c['err_m_at_5s']}m | 10s {c['err_m_at_10s']}m | "
            f"final {c['err_m_final']}m | breach@{c['first_breach_of_2m_s']}s | "
            f"scale {c['m_per_px_at_median_depth']} m/px (median box {c['box_h_px_median']}px) | {c['verdict']}"
        )

    (Path(__file__).parent / "verification.json").write_text(json.dumps(result, indent=2))
    print("\nwrote reports/verification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
