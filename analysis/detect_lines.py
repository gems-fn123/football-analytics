"""Find pitch markings in a frame: white line segments and the centre circle.

Correspondences for calibration must be *precise*. Reading pixel coordinates off an
image by eye is good to maybe five pixels, which at ~0.05 m/px is a quarter-metre of
error before anything else goes wrong. So the geometry is fitted from the image
instead: lines are detected and merged, and their intersections are computed
analytically. A human (or the caller) only has to say which line is which.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Hue band kept tight: the crowd in these stadiums wears a lot of teal (hue ~90),
# which a loose upper bound pulls straight into the "grass" class.
GRASS_HUE = (35, 80)
GRASS_MIN_SAT = 60
GRASS_MIN_VAL = 40


def _keep_elongated(mask: np.ndarray, min_len: float = 22.0, min_ratio: float = 3.0) -> np.ndarray:
    """Drop blobs that are not line-like, judged by PCA on their pixel coordinates."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), 8
    )
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 12:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.column_stack([xs, ys]).astype(float)
        pts -= pts.mean(0)
        if len(pts) < 8:
            continue
        ev = np.linalg.svd(pts, compute_uv=False)
        major = float(ev[0]) / np.sqrt(len(pts))
        minor = float(ev[1]) / np.sqrt(len(pts)) + 1e-6
        length = float(np.ptp(pts @ (np.linalg.svd(pts, full_matrices=False)[2][0])))
        if length >= min_len and major / minor >= min_ratio:
            out[lab == i] = 255
    return out


def field_mask(bgr: np.ndarray) -> np.ndarray:
    """The playing surface: largest connected grass region, holes filled.

    Taking the largest connected component is what keeps the stands out - the crowd
    contains plenty of green, but it is not contiguous with the pitch.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    grass = (
        (hsv[..., 0] >= GRASS_HUE[0])
        & (hsv[..., 0] <= GRASS_HUE[1])
        & (hsv[..., 1] >= GRASS_MIN_SAT)
        & (hsv[..., 2] >= GRASS_MIN_VAL)
    ).astype(np.uint8)
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grass, 8)
    if n <= 1:
        return grass * 255
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    field = (lab == biggest).astype(np.uint8)
    # Fill interior holes (players, painted lines) by contour fill rather than a large
    # closing: a big kernel bridges the touchline into the stands, which is exactly how
    # the crowd ends up classified as pitch.
    cnts, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(field)
    if cnts:
        cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    return filled * 255


def line_mask(bgr: np.ndarray, player_boxes: np.ndarray | None = None) -> np.ndarray:
    """White markings inside the playing surface, with players removed.

    Player kits are the dominant false positive: a white shirt is a bright,
    desaturated blob sitting on grass, which is exactly the signature of a painted
    line. The detector already knows where the players are, so they are cut out
    rather than fought with.
    """
    field = (field_mask(bgr) > 0).astype(np.uint8)
    # Erode hard: the grass/hoarding boundary is a strong bright edge that otherwise
    # dominates every line fit, and nothing useful lives in the outermost few pixels.
    field = cv2.erode(field, np.ones((11, 11), np.uint8))

    # A painted line is a *thin bright ridge on a locally uniform background*, which is
    # precisely what a white top-hat isolates. Absolute colour thresholds fail here:
    # sunlit grass is brighter than a shaded line, and the lines carry a green cast.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    lch = cv2.GaussianBlur(lab[..., 0], (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    tophat = cv2.morphologyEx(lch, cv2.MORPH_TOPHAT, k)

    inside = tophat[field > 0]
    if inside.size < 100:
        return np.zeros(field.shape, np.uint8)
    # An ABSOLUTE ridge threshold, not a percentile. A percentile always returns the
    # same fraction of the field as "line" whether or not any line is present, so the
    # mask fills up with mowing stripes and shadow edges and the downstream fit has no
    # way to tell good evidence from bad.
    # Robust, noise-adaptive: median + 4 MAD of the in-field ridge response. This adapts
    # to how bright a given stadium paints its lines instead of assuming a fixed share
    # of the pitch is painted, and gives comparable pixel counts across very different
    # clips (3.8k / 4.8k / 4.3k on the three tested here).
    med = float(np.median(inside))
    mad = float(np.median(np.abs(inside - med))) * 1.4826
    thr = max(4.0, med + 4.0 * mad)
    m = ((tophat >= thr) & (field > 0)).astype(np.uint8) * 255

    # Keep only elongated structures. A painted line is long and thin; mown grass
    # texture and shadow speckle are neither, and this is what separates them.
    m = _keep_elongated(m)

    if player_boxes is not None and len(player_boxes):
        for x1, y1, x2, y2 in player_boxes:
            pad_w = 0.35 * (x2 - x1)
            pad_h = 0.12 * (y2 - y1)
            cv2.rectangle(
                m,
                (int(x1 - pad_w), int(y1 - pad_h)),
                (int(x2 + pad_w), int(y2 + pad_h)),
                0,
                -1,
            )
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return m


def segments(mask: np.ndarray, min_len: int = 40) -> np.ndarray:
    ls = cv2.HoughLinesP(
        mask, 1, np.pi / 360, threshold=45, minLineLength=min_len, maxLineGap=14
    )
    return np.zeros((0, 4)) if ls is None else ls[:, 0, :].astype(float)


def _to_polar(seg: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = seg
    th = np.arctan2(y2 - y1, x2 - x1)
    th_n = (th + np.pi / 2) % np.pi  # normal direction
    rho = x1 * np.cos(th_n) + y1 * np.sin(th_n)
    return th_n, rho


def merge(segs: np.ndarray, ang_tol=np.deg2rad(3.0), rho_tol=14.0) -> list[dict]:
    """Group near-collinear segments and refit one line per group (total least squares)."""
    groups: list[list[np.ndarray]] = []
    keys: list[tuple[float, float]] = []
    for s in segs:
        th, rho = _to_polar(s)
        placed = False
        for i, (kth, krho) in enumerate(keys):
            dth = abs((th - kth + np.pi / 2) % np.pi - np.pi / 2)
            if dth < ang_tol and abs(rho - krho) < rho_tol:
                groups[i].append(s)
                n = len(groups[i])
                keys[i] = (kth + (th - kth) / n, krho + (rho - krho) / n)
                placed = True
                break
        if not placed:
            groups.append([s])
            keys.append((th, rho))

    out = []
    for g in groups:
        pts = np.array([[s[0], s[1]] for s in g] + [[s[2], s[3]] for s in g])
        length = sum(float(np.hypot(s[2] - s[0], s[3] - s[1])) for s in g)
        c = pts.mean(axis=0)
        u, s_, vt = np.linalg.svd(pts - c)
        d = vt[0]  # unit direction
        t = (pts - c) @ d
        p1, p2 = c + d * t.min(), c + d * t.max()
        out.append(
            {
                "p1": p1,
                "p2": p2,
                "dir": d,
                "centroid": c,
                "support": length,
                "n_seg": len(g),
                "angle_deg": float(np.rad2deg(np.arctan2(d[1], d[0])) % 180),
            }
        )
    out.sort(key=lambda r: -r["support"])
    return out


def intersect(a: dict, b: dict) -> np.ndarray | None:
    """Intersection of two fitted lines, or None if near-parallel."""
    d1, d2 = a["dir"], b["dir"]
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, b["centroid"] - a["centroid"])
    return a["centroid"] + d1 * t[0]


def fit_ellipse(mask: np.ndarray, lines: list[dict], img_shape) -> dict | None:
    """Fit the centre circle: the largest blob of line pixels that is not a line."""
    work = mask.copy()
    for ln in lines[:12]:
        cv2.line(work, tuple(ln["p1"].astype(int)), tuple(ln["p2"].astype(int)), 0, 9)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    for c in cnts:
        if len(c) < 40:
            continue
        try:
            e = cv2.fitEllipse(c)
        except cv2.error:
            continue
        (ex, ey), (MA, ma), ang = e
        if MA < 25 or ma < 6:
            continue
        peri = cv2.arcLength(c, False)
        score = peri
        if best is None or score > best["score"]:
            best = {"centre": (ex, ey), "axes": (MA, ma), "angle": ang, "score": score,
                    "n_pts": len(c)}
    return best


def annotate(bgr: np.ndarray, lines: list[dict], ell: dict | None, scale=2.0) -> np.ndarray:
    img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    palette = [(0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 200, 0), (255, 0, 200),
               (255, 255, 0), (128, 0, 255), (0, 128, 255), (200, 255, 128), (255, 128, 128)]
    for i, ln in enumerate(lines[:10]):
        col = palette[i % len(palette)]
        p1 = (ln["p1"] * scale).astype(int)
        p2 = (ln["p2"] * scale).astype(int)
        cv2.line(img, tuple(p1), tuple(p2), col, 2)
        mid = ((p1 + p2) / 2).astype(int)
        cv2.putText(img, f"L{i}", tuple(mid), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    if ell:
        c = (int(ell["centre"][0] * scale), int(ell["centre"][1] * scale))
        ax = (int(ell["axes"][0] * scale / 2), int(ell["axes"][1] * scale / 2))
        cv2.ellipse(img, c, ax, ell["angle"], 0, 360, (255, 255, 255), 2)
        cv2.putText(img, "ELL", (c[0] + 6, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return img


def player_boxes_for(tracks_px: str | Path, frame_idx: int) -> np.ndarray:
    import pandas as pd

    df = pd.read_parquet(tracks_px)
    f = df[(df["frame"] == frame_idx) & (df["cls"] != "ball")]
    if not len(f):
        return np.zeros((0, 4))
    return f[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)


def analyse(
    video: str, frame_idx: int, out_png: str | None = None, tracks_px: str | None = None
) -> dict:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx} of {video}")
    boxes = player_boxes_for(tracks_px, frame_idx) if tracks_px else None
    m = line_mask(img, boxes)
    lines = merge(segments(m))
    ell = fit_ellipse(m, lines, img.shape)
    if out_png:
        cv2.imwrite(out_png, annotate(img, lines, ell))
    return {"image": img, "mask": m, "lines": lines, "ellipse": ell, "field": field_mask(img)}


if __name__ == "__main__":
    video, fidx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    tpx = sys.argv[4] if len(sys.argv) > 4 else None
    r = analyse(video, fidx, out, tpx)
    print(f"{Path(video).name} frame {fidx}: {len(r['lines'])} merged lines")
    for i, ln in enumerate(r["lines"][:10]):
        print(
            f"  L{i}: angle {ln['angle_deg']:6.1f} deg  support {ln['support']:7.1f}px  "
            f"segs {ln['n_seg']:3d}  p1 ({ln['p1'][0]:6.1f},{ln['p1'][1]:6.1f})  "
            f"p2 ({ln['p2'][0]:6.1f},{ln['p2'][1]:6.1f})"
        )
    if r["ellipse"]:
        e = r["ellipse"]
        print(f"  ELLIPSE centre ({e['centre'][0]:.1f},{e['centre'][1]:.1f}) "
              f"axes ({e['axes'][0]:.1f},{e['axes'][1]:.1f}) angle {e['angle']:.1f} pts {e['n_pts']}")
    print("wrote", out)
