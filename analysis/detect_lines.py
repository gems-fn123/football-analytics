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
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conics  # noqa: E402

# Hue band kept tight: the crowd in these stadiums wears a lot of teal (hue ~90),
# which a loose upper bound pulls straight into the "grass" class.
GRASS_HUE = (35, 80)
GRASS_MIN_SAT = 60
GRASS_MIN_VAL = 40


def ridge_response(bgr: np.ndarray) -> np.ndarray:
    """Thin-bright-ridge response: a white top-hat on the Lab lightness channel.

    Absolute colour thresholds fail here - sunlit grass is brighter than a shaded line,
    and the lines carry a green cast - whereas a top-hat responds to the *shape* of the
    intensity profile regardless of illumination.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    lch = cv2.GaussianBlur(lab[..., 0], (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    return cv2.morphologyEx(lch, cv2.MORPH_TOPHAT, k)


def static_overlay_mask(
    video: str | Path, n_frames: int = 24, persist: float = 0.90, cache_dir: Path | None = None
) -> np.ndarray | None:
    """Pixels that stay bright and thin in *image* space while the camera moves.

    A burnt-in broadcast graphic - a stock watermark, a score bug, a channel logo - is
    fixed to the frame. Everything the pitch model cares about is fixed to the ground. So
    the two separate on persistence alone, with no need to know what the graphic says or
    looks like: as the camera pans, markings sweep across the image and the graphic does
    not. On these clips the iStock watermark sits directly over the centre circle and was
    dominating the conic fit, which is what made this worth doing properly rather than
    special-casing.

    Returns None when the camera barely moves, because then the test cannot distinguish
    an overlay from a marking and claiming otherwise would delete real evidence.

    Measured cost on clip0, against its verified homography: this removes ~3 000 px of
    watermark but takes centre-circle recall from 58% to 42% and the touchline from 17%
    to 3%, so it is **off by default**. The reason is a real limitation rather than a
    tuning problem: clip0 pans horizontally, and a structure parallel to the pan - the
    touchline above all - stays at the same image height throughout, so it looks exactly
    as persistent as a burnt-in graphic. Testing "fixed to the ground" properly means
    warping the frames into a common ground frame first, not testing "fixed to the image"
    and hoping the camera moved the right way.

    Two details are load-bearing. The field mask is applied *per frame*, otherwise the
    accumulator fills with crowd - bright, textured, and the most persistent thing in
    any stadium. And persistence is judged over the frames in which a pixel was actually
    inside the field, not over all sampled frames, since a pixel the pitch only covers
    half the time cannot respond in the other half.
    """
    video = Path(video)
    cache = (cache_dir / f"overlay_{video.stem}.png") if cache_dir else None
    if cache is not None and cache.exists():
        return cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 1, 0), min(n_frames, max(total, 1))).astype(int)
    acc = seen = None
    n_read, first, last = 0, None, None
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        fld = cv2.erode((field_mask(img) > 0).astype(np.uint8), np.ones((11, 11), np.uint8))
        r = ridge_response(img)
        ins = r[fld > 0]
        if ins.size < 100:
            continue
        med = float(np.median(ins))
        mad = float(np.median(np.abs(ins - med))) * 1.4826
        hit = ((r >= max(4.0, med + 4.0 * mad)) & (fld > 0)).astype(np.uint16)
        acc = hit.copy() if acc is None else acc + hit
        seen = fld.astype(np.uint16) if seen is None else seen + fld
        first = img if first is None else first
        last, n_read = img, n_read + 1
    cap.release()
    if acc is None or n_read < 6:
        return None

    # Did the camera move enough for persistence to mean anything? Phase correlation
    # between the first and last sampled frame answers it in one number.
    g0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(g0, g1)
    if float(np.hypot(dx, dy)) < 40.0:
        return None

    enough = seen >= max(6, int(0.5 * n_read))
    persistent = enough & (acc >= persist * np.maximum(seen, 1))
    out = cv2.dilate(persistent.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cache), out)
    return out


def _hysteresis(resp: np.ndarray, field: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """Threshold at `hi`, then grow through connected pixels above `lo`.

    A single threshold cannot work on a pitch marking, because one marking is not one
    brightness: measured along clip0's known centre circle the ridge response runs from
    3 to 67, so any level that excludes grass texture also chops the circle into arcs too
    short to survive a length filter. Growing from confident seeds through weaker
    connected evidence recovers the whole marking while still refusing to start anywhere
    the evidence is only weak - which is what a plain low threshold would do.
    """
    strong = (resp >= hi) & (field > 0)
    weak = ((resp >= lo) & (field > 0)).astype(np.uint8)
    if not strong.any():
        return np.zeros(resp.shape, np.uint8)
    n, lab = cv2.connectedComponents(weak, 8)
    keep = np.unique(lab[strong])
    keep = keep[keep > 0]
    if not len(keep):
        return np.zeros(resp.shape, np.uint8)
    flags = np.zeros(n, bool)
    flags[keep] = True
    return (flags[lab]).astype(np.uint8) * 255


def _keep_thin_and_long(
    mask: np.ndarray, max_stroke: float = 9.0, min_len: float = 25.0
) -> np.ndarray:
    """Keep components that are *thin and long*, whatever shape they run in.

    The previous test measured PCA elongation - the ratio of a component's major to minor
    spread - and demanded at least 3:1. That is a prior on straightness, not on being a
    marking, and it silently deleted the centre circle: a closed rim is as wide as it is
    tall, so it scores near 1:1 and never survived. Measuring the response along the known
    circle in clip0 showed 57% of it clearing the intensity threshold and none of it
    reaching the mask, which located the loss here rather than in the thresholding.

    Thinness is twice the 95th percentile of the component's distance transform, and
    length is area divided by that, which estimates centreline length for a curve as well
    as for a straight run. The percentile rather than the maximum matters: a marking is
    one connected component from end to end, so judging it by its single widest point
    lets one fat patch - a line crossing another, or a scuff - delete the entire line.
    """
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    out = np.zeros_like(mask)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 12:
            continue
        sub = lab[y : y + bh, x : x + bw] == i
        stroke = 2.0 * float(np.percentile(dt[y : y + bh, x : x + bw][sub], 95))
        if stroke > max_stroke:
            continue
        if area / max(stroke, 1.0) < min_len:
            continue
        out[y : y + bh, x : x + bw][sub] = 255
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


def line_mask(
    bgr: np.ndarray,
    player_boxes: np.ndarray | None = None,
    overlay: np.ndarray | None = None,
) -> np.ndarray:
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

    # A painted line is a thin bright ridge on a locally uniform background, which is
    # precisely what a white top-hat isolates.
    tophat = ridge_response(bgr)

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
    m = _hysteresis(tophat, field, max(4.0, med + 4.0 * mad), max(2.0, med + 2.0 * mad))

    # Keep only thin, long structures. A painted marking is thin and runs a long way;
    # mown grass texture and shadow speckle are neither. Thinness rather than
    # straightness, so that the centre circle survives.
    m = _keep_thin_and_long(m)

    # Burnt-in graphics come out *last*, after growth and filtering. Removing them from
    # the field beforehand cuts long markings into fragments that the length filter then
    # deletes, so a 20% loss of pixels turned into a 65% loss of evidence. Subtracting at
    # the end leaves the connectivity that faint markings depend on intact.
    if overlay is not None:
        m[overlay > 0] = 0

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


def bow(line: dict, mask: np.ndarray, halfwidth: float = 4.0, bins: int = 8) -> float:
    """Peak sideways deviation of the evidence under a fitted line, in pixels.

    Hough finds straight segments, and the bottom of a projected centre circle is nearly
    straight over a long span - so the circle is repeatedly detected as a *line*. That
    matters twice over: erasing the detected lines removed 97% of clip0's centre circle,
    and crediting a conic for pixels that a straight line already explains let a conic
    draped along the halfway line outscore the true rim.

    A chord of an arc bows away from its own best-fit line in a smooth, single-signed
    curve, while a painted line does not. Measuring the deviation in bins along the line
    detects that, and needs no prior knowledge of which markings are present.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    p = np.column_stack([xs, ys]).astype(float) - line["centroid"]
    d = line["dir"]
    t = p @ d
    s = p @ np.array([-d[1], d[0]])
    ext = float(np.hypot(*(line["p2"] - line["p1"]))) / 2.0
    sel = (np.abs(s) <= halfwidth) & (np.abs(t) <= ext)
    if sel.sum() < 30:
        return 0.0
    t, s = t[sel], s[sel]
    edges = np.linspace(t.min(), t.max(), bins + 1)
    means = [s[(t >= edges[i]) & (t <= edges[i + 1])].mean() for i in range(bins)
             if ((t >= edges[i]) & (t <= edges[i + 1])).sum() >= 5]
    return 0.0 if len(means) < 4 else float(np.ptp(means))


def straight_lines(lines: list[dict], mask: np.ndarray, max_bow: float = 1.5) -> list[dict]:
    """The merged lines that are genuinely straight, not chords of a circle.

    The threshold is deliberately strict, because the two errors are not symmetric:
    treating a real line as curved merely leaves some evidence uncredited, whereas
    treating an arc as a line discards the one landmark that makes registration possible.
    """
    out = []
    for ln in lines:
        b = bow(ln, mask)
        ln = {**ln, "bow": b}
        if b <= max_bow:
            out.append(ln)
    return out


def line_pixel_mask(lines: list[dict], shape, thickness: int = 7, margin: float = 0.15):
    """Pixels already accounted for by straight markings.

    Bounded by each line's detected extent rather than extended across the frame: a line
    is only evidence where it was actually seen, and extending it discounts unrelated
    pixels that happen to be collinear with it somewhere else.
    """
    m = np.zeros(shape[:2], np.uint8)
    for ln in lines:
        p1, p2 = ln["p1"], ln["p2"]
        d = (p2 - p1) * margin
        cv2.line(m, tuple((p1 - d).astype(int)), tuple((p2 + d).astype(int)), 255, thickness)
    return m


def _erase_lines(
    mask: np.ndarray, lines: list[dict], thickness: int = 11, mode: str = "extend"
) -> np.ndarray:
    """Blank out the fitted straight lines so only curved evidence remains.

    `mode` controls how far the erasure reaches: "extend" past the detected support,
    "extent" only across it, or "none" to leave the mask alone.
    """
    if mode == "none":
        return mask.copy()
    work = mask.copy()
    h, w = mask.shape
    reach = float(2 * (h + w))
    for ln in lines[:14]:
        if mode == "extend":
            c, d = ln["centroid"], ln["dir"]
            a, b = (c - d * reach).astype(int), (c + d * reach).astype(int)
        else:
            a, b = ln["p1"].astype(int), ln["p2"].astype(int)
        cv2.line(work, tuple(a), tuple(b), 0, thickness)
    return work


TOL_SCHEDULE = (16.0, 11.0, 7.0, 5.0, 3.5, 3.0, 3.0)


def _grow_and_refit(C, pts, tols=TOL_SCHEDULE):
    """Alternate between collecting inliers and refitting, seeded from a rough conic.

    The tolerance is annealed from wide to tight. A seed fitted to one fragment of the
    rim is biased - the algebraic fit of a short arc under-estimates its size - so at a
    tight tolerance it can only ever re-find the fragment it came from. Starting wide
    lets it reach the rest of the rim, and each refit pulls the estimate closer, so the
    tolerance can then be closed without losing what was gained.
    """
    inl = None
    for tol in tols:
        d = conics.sampson(C, pts)
        sel = d < tol
        if sel.sum() < conics.MIN_POINTS:
            return (C, inl) if inl is not None else (None, None)
        nxt = conics.fit_conic(pts[sel])
        if nxt is None:
            return C, pts[sel]
        C, inl = nxt, pts[sel]
    return C, inl


def fit_ellipses(mask: np.ndarray, lines: list[dict], img_shape, seed: int = 0,
                 top_k: int = 8, erase: str = "none", erase_px: int = 11) -> list[dict]:
    """Candidate images of a pitch circle, best first.

    Deliberately returns *candidates* rather than an answer. Measured against clip0's
    verified geometry, no local score reliably picks the centre circle out: only 58% of
    its rim clears the intensity threshold, so a conic sweeping across several markings
    genuinely explains more evidence than the true circle does (16.7 against 6.2 here).
    That is not a tuning failure, it is the evidence being ambiguous.

    The pose fit resolves it instead. A candidate bound to the wrong circle produces a
    camera that fails to explain the rest of the pitch, and the full-model objective sees
    that immediately - so the detector proposes and the registration disposes.

    Each candidate carries the supporting pixels, and downstream fitting uses those
    rather than the conic: a conic fitted to a partial arc extrapolates badly, while the
    pixels it selects are still the right pixels.
    """
    h, w = img_shape[:2]
    # Fit on *all* the evidence. Erasing the detected lines first is what destroyed the
    # circle, because its flattest arcs are detected as lines; instead the straight
    # markings are identified and discounted at scoring time, below.
    straights = straight_lines(lines, mask)
    line_px = line_pixel_mask(straights, mask.shape)
    curved_mask = mask.copy()
    curved_mask[line_px > 0] = 0

    work = _erase_lines(mask, lines, erase_px, erase)
    ys, xs = np.nonzero(work)
    if len(xs) < 60:
        return []
    all_pts = np.column_stack([xs, ys]).astype(float)
    rng = np.random.default_rng(seed)
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(work, 8)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:14] + 1

    comps: list[np.ndarray] = []
    for i in order:
        if stats[i, cv2.CC_STAT_AREA] < 25:
            continue
        cy_, cx_ = np.nonzero(lab == i)
        p = np.column_stack([cx_, cy_]).astype(float)
        if len(p) > 400:
            p = p[rng.choice(len(p), 400, replace=False)]
        comps.append(p)

    seeds: list[np.ndarray] = list(comps)
    # A rim broken by players standing on it is several components, and no single one of
    # them describes the whole conic well enough to grow from. Pairs of components span
    # far more of the rim, which is what makes the seed conic close enough to be useful.
    for i in range(len(comps)):
        for j in range(i + 1, min(i + 6, len(comps))):
            seeds.append(np.vstack([comps[i], comps[j]]))
    # Blind RANSAC, sampled *locally*: six points drawn uniformly from the whole residual
    # set come from the same structure about as often as never (at a 25% inlier rate,
    # 1 draw in 6 500), whereas six points drawn from one neighbourhood usually do.
    tree = cKDTree(all_pts)
    for _ in range(500):
        c = all_pts[rng.integers(len(all_pts))]
        near = tree.query_ball_point(c, r=180.0)
        if len(near) >= 8:
            seeds.append(all_pts[rng.choice(near, 8, replace=False)])

    found: list[dict] = []
    for p in seeds:
        C0 = conics.fit_conic(p)
        if C0 is None:
            continue
        C, inl = _grow_and_refit(C0, all_pts)
        if C is None or inl is None:
            continue
        # Credit only evidence a straight marking does not already explain. Without this
        # a conic running along the halfway line for part of its rim outscored the true
        # centre circle, because by every local measure it really was the better
        # explanation of the pixels.
        keep = line_px[inl[:, 1].astype(int).clip(0, h - 1),
                       inl[:, 0].astype(int).clip(0, w - 1)] == 0
        curved = inl[keep]
        m = conics.validate(C, curved, (h, w))
        if m is None:
            continue
        # Both directions of agreement. Support alone (evidence spread around the rim) is
        # satisfied by a conic draped over a line; rim precision alone (rim sitting on
        # evidence) is satisfied by a small conic hiding inside a thick marking.
        m["rim_precision"] = conics.rim_precision(C, curved_mask)
        score = m["support"] * m["rim_precision"] * float(np.sqrt(min(m["n_pts"], 900)))
        found.append({"conic": C, "points": curved, "score": float(score), **m})

    found.sort(key=lambda r: -r["score"])
    out: list[dict] = []
    for c in found:
        # Distinct candidates only, else the list fills with near-copies of one answer
        # and the pose fit never gets a second hypothesis to consider.
        if any(
            float(np.hypot(c["centre"][0] - o["centre"][0], c["centre"][1] - o["centre"][1])) < 25.0
            and abs(c["axes"][0] - o["axes"][0]) < 0.25 * max(o["axes"][0], 1.0)
            for o in out
        ):
            continue
        out.append(c)
        if len(out) >= top_k:
            break
    return out


def fit_ellipse(mask: np.ndarray, lines: list[dict], img_shape, seed: int = 0,
                erase: str = "none", erase_px: int = 11) -> dict | None:
    """Highest-scoring circle candidate, or None. See `fit_ellipses` for the caveats."""
    c = fit_ellipses(mask, lines, img_shape, seed, 1, erase, erase_px)
    return c[0] if c else None


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
    video: str,
    frame_idx: int,
    out_png: str | None = None,
    tracks_px: str | None = None,
    drop_overlays: bool = False,
) -> dict:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx} of {video}")
    boxes = player_boxes_for(tracks_px, frame_idx) if tracks_px else None
    ov = (
        static_overlay_mask(video, cache_dir=Path(__file__).resolve().parent / "calib")
        if drop_overlays
        else None
    )
    m = line_mask(img, boxes, ov)
    lines = merge(segments(m))
    cands = fit_ellipses(m, lines, img.shape)
    if out_png:
        cv2.imwrite(out_png, annotate(img, lines, cands[0] if cands else None))
    return {"image": img, "mask": m, "lines": lines,
            "ellipse": cands[0] if cands else None, "ellipses": cands,
            "straight": straight_lines(lines, m), "field": field_mask(img), "overlay": ov}


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
              f"axes ({e['axes'][0]:.1f},{e['axes'][1]:.1f}) angle {e['angle']:.1f} "
              f"support {e['support']:.2f} pts {e['n_pts']}")
    else:
        print("  ELLIPSE none")
    print("wrote", out)
