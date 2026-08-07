"""Can appearance re-ID rejoin fragmented tracks at this resolution?

Scored against C-S1 / C-S2 in reports/02_preregistration_metrics.md.

Order matters and is not arbitrary: split on physics first, then rejoin on appearance.
A ByteTrack id that teleports contains two different players, so merging anything onto
it compounds the error; analysis/kinematics.py cuts those first. Stitching then merges
fragments that are separated in time, which is the complementary failure.

Before any merge is believed, the embedding itself is validated. Players here are ~36 px
tall and OSNet expects 256x128, so the crops are heavily upscaled and the embedding may
simply not be discriminative. That is testable without any ground-truth labels:

    positive pairs - two crops from the SAME segment: same player by construction.
    negative pairs - crops from two segments that OVERLAP IN TIME: provably different
                     players, since one person cannot be in two places in one frame.

If those two similarity distributions overlap, no threshold can separate players and
stitching cannot work at this resolution. That is a property of the footage, and it is
better to establish it than to ship merges that look plausible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from footy.reid.embedder import Embedder  # noqa: E402
from footy.reid.stitch import stitch_tracks  # noqa: E402
from kinematics import (  # noqa: E402
    add_kinematics,
    estimate_position_noise,
    split_on_impossible_motion,
    summarise,
)

ROOT = HERE.parent
PROC = ROOT / "data/processed/clip0"
VIDEO = str(ROOT / "data/raw/smoke_clip.mp4")
WEIGHTS = str(ROOT / "models/weights/osnet_x0_25.pt")
MAX_CROPS = 6
SPEED_CAP_MS = 11.0
MEDIAN_PERSONS_PER_FRAME = 15.0     # from reports/verification.json, clip0


def collect_crops(df: pd.DataFrame, max_crops: int = MAX_CROPS):
    """segment id -> list of BGR crops, sampled across the segment's life."""
    want: dict[int, list[int]] = {}
    for tid, g in df.groupby("track_id"):
        frames = sorted(g["frame"].unique())
        idx = np.linspace(0, len(frames) - 1, min(max_crops, len(frames))).astype(int)
        want[int(tid)] = [int(frames[i]) for i in idx]
    by_frame: dict[int, list[int]] = {}
    for tid, frames in want.items():
        for f in frames:
            by_frame.setdefault(f, []).append(tid)

    crops: dict[int, list[np.ndarray]] = {t: [] for t in want}
    cap = cv2.VideoCapture(VIDEO)
    lookup = df.set_index(["frame", "track_id"])
    for f in sorted(by_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        for tid in by_frame[f]:
            try:
                row = lookup.loc[(f, tid)]
            except KeyError:
                continue
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            x1, y1 = max(int(row["x1"]), 0), max(int(row["y1"]), 0)
            x2, y2 = min(int(row["x2"]), w), min(int(row["y2"]), h)
            if x2 - x1 < 4 or y2 - y1 < 8:
                continue
            crops[tid].append(img[y1:y2, x1:x2].copy())
    cap.release()
    return crops


def main() -> int:
    df = pd.read_parquet(PROC / "tracks_m_tracked.parquet")
    players = df[(df["cls"] == "player") & (df["track_id"] >= 0)].dropna(subset=["x_m", "y_m"])
    n_bytetrack = players["track_id"].nunique()

    sigma = estimate_position_noise(players)
    seg = split_on_impossible_motion(players, sigma)
    n_segments = seg["track_id"].nunique()
    print(f"clip0: {n_bytetrack} ByteTrack ids -> {n_segments} physics segments")

    box_h = (seg["y2"] - seg["y1"])
    print(f"  player box height: median {box_h.median():.0f} px, p90 {box_h.quantile(.9):.0f} px "
          f"(OSNet input is 256x128)")

    emb = Embedder(WEIGHTS)
    print(f"  OSNet weights loaded: {emb.loaded_fraction:.1%} of parameters matched")
    if emb.loaded_fraction < 0.5:
        print("  refusing to continue: weights did not load")
        return 1

    print("\ncollecting crops...")
    crops = collect_crops(seg)
    usable = {t: c for t, c in crops.items() if len(c) >= 2}
    print(f"  {len(usable)} segments with >=2 crops")

    # ---------- validity: is the embedding discriminative at this size? ----------
    per_crop = {t: emb.embed(c) for t, c in usable.items()}
    spans = {int(t): (int(g["frame"].min()), int(g["frame"].max()))
             for t, g in seg.groupby("track_id")}

    pos = []
    for t, E in per_crop.items():
        for i in range(len(E)):
            for j in range(i + 1, len(E)):
                pos.append(float(E[i] @ E[j]))
    rng = np.random.default_rng(0)
    tids = sorted(per_crop)
    neg = []
    for _ in range(4000):
        a, b = rng.choice(tids, 2, replace=False)
        sa, sb = spans[int(a)], spans[int(b)]
        if sa[1] < sb[0] or sb[1] < sa[0]:
            continue  # not simultaneous: could be the same player, not a valid negative
        neg.append(float(per_crop[a][rng.integers(len(per_crop[a]))]
                         @ per_crop[b][rng.integers(len(per_crop[b]))]))
    pos, neg = np.array(pos), np.array(neg)
    print(f"\nembedding validity ({len(pos)} same-player pairs, {len(neg)} "
          f"provably-different pairs):")
    print(f"  same player      median {np.median(pos):.3f}   p10 {np.percentile(pos,10):.3f}")
    print(f"  different player median {np.median(neg):.3f}   p90 {np.percentile(neg,90):.3f}")
    overlap = float((neg > np.percentile(pos, 10)).mean())
    print(f"  fraction of DIFFERENT-player pairs scoring above the 10th percentile of "
          f"SAME-player pairs: {overlap:.1%}")
    # Separability: how well does any single threshold do?
    best_acc, best_thr = 0.0, None
    for thr in np.linspace(-0.2, 1.0, 121):
        acc = 0.5 * ((pos >= thr).mean() + (neg < thr).mean())
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    print(f"  best achievable balanced accuracy: {best_acc:.1%} at threshold {best_thr:.2f}")
    print("  (50% = the embedding carries no information; 100% = perfectly separable)")

    # ---------- run the stitch ----------
    tracks = {}
    for t, E in per_crop.items():
        e = E.mean(axis=0)
        tracks[int(t)] = {"start": spans[int(t)][0], "end": spans[int(t)][1],
                          "embedding": e / max(np.linalg.norm(e), 1e-9)}
    mapping = stitch_tracks(tracks)
    n_merged_groups = len(set(mapping.values()))
    n_after = n_segments - (len(tracks) - n_merged_groups)
    print(f"\nstitching: {len(tracks)} embeddable segments -> {n_merged_groups} groups "
          f"({len(tracks) - n_merged_groups} merges)")

    stitched = seg.copy()
    stitched["track_id"] = stitched["track_id"].map(lambda t: mapping.get(int(t), int(t)))

    # ---------- C-S2: a false merge is provable ----------
    dup = stitched.groupby(["track_id", "frame"]).size()
    violations = int((dup > 1).sum())
    print(f"\nC-S2  merged tracks containing two detections in one frame: {violations}")

    # ---------- C-S1 ----------
    n_ids = stitched["track_id"].nunique()
    cap_ids = 5 * MEDIAN_PERSONS_PER_FRAME
    print(f"C-S1  distinct ids after stitching: {n_ids}  (cap {cap_ids:.0f})")

    crit = {
        "C-S1 ids <= 5x median persons": {"pass": bool(n_ids <= cap_ids),
                                          "value": int(n_ids), "cap": int(cap_ids)},
        "C-S2 no same-frame duplicates": {"pass": violations == 0, "value": violations},
    }

    # ---------- effect on the physical criteria ----------
    kin, diag = add_kinematics(stitched)
    kin = kin.dropna(subset=["speed_ms"])
    over = float((kin["speed_ms"] > SPEED_CAP_MS).mean()) if len(kin) else float("nan")
    summ = summarise(kin)
    summ = summ[summ["minutes"] > 0]
    rate = float((summ["distance_m"] / 1000.0 * (90.0 / summ["minutes"])).median()) if len(summ) else float("nan")
    print(f"\neffect on physical criteria:")
    print(f"  C-P1 speeds over {SPEED_CAP_MS} m/s: {over:.2%}  (bar <=1%)")
    print(f"  C-P2 median rate: {rate:.1f} km/90  (bar 9-12)")
    print(f"  median track life: {summ['minutes'].median()*60:.1f} s" if len(summ) else "")

    print("\ncriteria:")
    for k, v in crit.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}  = {v['value']}")

    out = {
        "n_bytetrack_ids": int(n_bytetrack),
        "n_physics_segments": int(n_segments),
        "n_embeddable": len(tracks),
        "n_merges": int(len(tracks) - n_merged_groups),
        "n_ids_after": int(n_ids),
        "osnet_loaded_fraction": round(float(emb.loaded_fraction), 4),
        "box_height_median_px": float(box_h.median()),
        "embedding_same_player_median": round(float(np.median(pos)), 4),
        "embedding_diff_player_median": round(float(np.median(neg)), 4),
        "embedding_best_balanced_accuracy": round(float(best_acc), 4),
        "embedding_best_threshold": round(float(best_thr), 3),
        "same_frame_violations": violations,
        "speed_over_cap_frac": round(over, 4),
        "median_km_per_90": round(rate, 3),
        "criteria": crit,
    }
    (ROOT / "reports/stitching.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/stitching.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
