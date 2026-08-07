"""Velocity from noisy tracked positions, with the window derived from the noise.

Why not `footy.analytics.physical.add_kinematics`: it smooths positions with a rolling
mean and then differences adjacent frames. Differencing over a single frame divides the
position noise by dt = 0.04 s, which amplifies it enormously. Measured on clip0:

    position noise             sigma ~= 0.21 m   (from the per-frame step distribution,
                                                  and consistent with propagating 2 px
                                                  of feet jitter through the homography)
    single-frame differencing  sigma_v = sigma*sqrt(2)/dt = 7.4 m/s

That is why 20.1% of raw per-frame steps imply speeds above 11 m/s, and why the median
"speed" came out at 4.09 m/s - far too high for a median footballer, because it is
mostly noise rather than motion.

The window is DERIVED, not swept. For an ordinary-least-squares line fit over N samples
spaced dt apart, the slope's standard error is

    sigma_v = sigma / (dt * sqrt(N*(N^2-1)/12))

Requiring sigma_v <= 0.5 m/s - so that the 11 m/s plausibility bound sits many sigma
away from noise rather than inside it - gives N*(N^2-1) >= 1323, hence N >= 12. N = 13
is used (odd, so the estimate is centred), giving sigma_v = 0.39 m/s.

The cost is honest and stated: a 0.52 s window cannot resolve accelerations faster than
that, so peak instantaneous speeds are slightly attenuated. For distance covered - which
integrates - this is the correct trade, and integrating a noise-inflated speed would
otherwise overestimate distance through pure rectification bias (|noise| is always
positive and never cancels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DT_DEFAULT = 0.04
TARGET_SIGMA_V_MS = 0.5


def derive_window(sigma_m: float, dt: float = DT_DEFAULT,
                  target_sigma_v: float = TARGET_SIGMA_V_MS) -> int:
    """Smallest odd N whose OLS slope error meets the target. See module docstring."""
    n = 3
    while n < 201:
        sigma_v = sigma_m / (dt * np.sqrt(n * (n * n - 1) / 12.0))
        if sigma_v <= target_sigma_v:
            return n
        n += 2
    return n


def estimate_position_noise(df: pd.DataFrame) -> float:
    """Position noise sigma, from the median consecutive-frame step.

    For a stationary target with independent per-frame noise sigma, the expected step
    length is sigma*sqrt(2)*sqrt(2/pi). The median step is dominated by players who are
    near-stationary, so inverting that relation is a conservative noise estimate - real
    motion only inflates it, which makes the derived window longer, not shorter.
    """
    g = df.sort_values(["track_id", "frame"]).groupby("track_id")
    step = np.hypot(g["x_m"].diff(), g["y_m"].diff())
    dt_frames = g["frame"].diff()
    step = step[dt_frames == 1].dropna()
    if not len(step):
        return float("nan")
    return float(step.median() / (np.sqrt(2) * np.sqrt(2 / np.pi)))


MAX_HUMAN_SPEED_MS = 12.0   # above the fastest recorded footballer; a hard physical bound


def split_on_impossible_motion(df: pd.DataFrame, sigma_m: float,
                               max_speed: float = MAX_HUMAN_SPEED_MS) -> pd.DataFrame:
    """Cut a track wherever it moves faster than a human can. Returns a new track_id column.

    A track that teleports is not one player - it is ByteTrack handing one id to two
    different people. Merging such a track's frames into one "player" corrupts distance
    and top speed, which is exactly the failure "nullable beats guessed" exists to stop.
    Splitting is the conservative response: it produces more, shorter, but internally
    honest tracks, and appearance re-ID can afterwards rejoin the ones that do belong
    together.

    The threshold is not the bare physical bound. Position noise alone produces steps of
    order sigma, so a cut at max_speed*dt would fire constantly on stationary players.
    The bound used is

        max_speed * dt + 3 * sigma * sqrt(2)

    i.e. a step must be impossible by more than 3 sigma of the two-sample step noise
    before the track is cut, so noise essentially never triggers a split.
    """
    out = df.sort_values(["track_id", "frame"]).copy()
    g = out.groupby("track_id")
    step = np.hypot(g["x_m"].diff(), g["y_m"].diff())
    dt = g["t_s"].diff()
    bound = max_speed * dt + 3.0 * sigma_m * np.sqrt(2.0)
    impossible = (step > bound) & step.notna() & dt.notna()
    # A new segment starts at each impossible transition and at each track boundary.
    new_track = g.cumcount().eq(0) | impossible
    out["track_id"] = new_track.cumsum().astype(int) - 1
    return out


def _fit_velocity(t: np.ndarray, x: np.ndarray, y: np.ndarray, n: int):
    """Centred OLS slope over a window of n samples; NaN where support is short."""
    half = n // 2
    vx = np.full(len(t), np.nan)
    vy = np.full(len(t), np.nan)
    for i in range(len(t)):
        lo, hi = max(0, i - half), min(len(t), i + half + 1)
        tt, xx, yy = t[lo:hi], x[lo:hi], y[lo:hi]
        ok = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(tt)
        # Require most of the window: a slope from 3 points spanning 0.5 s is not the
        # estimator whose error was derived above, and would silently be much noisier.
        if ok.sum() < max(5, int(0.6 * n)):
            continue
        tc = tt[ok] - tt[ok].mean()
        denom = float((tc * tc).sum())
        if denom < 1e-12:
            continue
        vx[i] = float((tc * (xx[ok] - xx[ok].mean())).sum() / denom)
        vy[i] = float((tc * (yy[ok] - yy[ok].mean())).sum() / denom)
    return vx, vy


def add_kinematics(tracks: pd.DataFrame, window: int | None = None,
                   sigma_m: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Add speed_ms / accel_ms2 by windowed regression. Returns (df, diagnostics)."""
    out = tracks.copy()
    out["speed_ms"] = np.nan
    out["accel_ms2"] = np.nan

    tracked = out[out["track_id"] >= 0]
    if tracked.empty:
        return out, {"window": None, "sigma_m": None}

    if sigma_m is None:
        sigma_m = estimate_position_noise(tracked.dropna(subset=["x_m", "y_m"]))
    if window is None:
        window = derive_window(sigma_m)
    sigma_v = sigma_m / (DT_DEFAULT * np.sqrt(window * (window**2 - 1) / 12.0))

    for tid, g in out[out["track_id"] >= 0].groupby("track_id"):
        g = g.sort_values("frame")
        idx = g.index
        # Window on FRAME NUMBER, not row position. A track with gaps (missed
        # detections, or positions refused by the spatial gate) has rows that are
        # adjacent in the table but far apart in time; windowing by row would quietly
        # fit a line across the gap and call the result a velocity.
        frames = g["frame"].to_numpy().astype(int)
        full = np.arange(frames.min(), frames.max() + 1)
        pos = {f: i for i, f in enumerate(full)}
        xs = np.full(len(full), np.nan)
        ys = np.full(len(full), np.nan)
        ts = full * DT_DEFAULT
        where = np.array([pos[f] for f in frames])
        xs[where] = g["x_m"].to_numpy(dtype=float)
        ys[where] = g["y_m"].to_numpy(dtype=float)
        vx_full, vy_full = _fit_velocity(ts, xs, ys, window)
        vx, vy = vx_full[where], vy_full[where]
        speed = np.hypot(vx, vy)
        out.loc[idx, "speed_ms"] = speed
        # Acceleration from the same regression output, differenced over the window
        # rather than per frame, for the same noise reason.
        acc = np.full(len(speed), np.nan)
        half = window // 2
        if len(speed) > window:
            acc[half:-half] = (speed[window - 1:] - speed[: -(window - 1)]) / (
                (window - 1) * DT_DEFAULT
            )
        out.loc[idx, "accel_ms2"] = acc

    return out, {
        "window_frames": int(window),
        "window_seconds": round(window * DT_DEFAULT, 3),
        "sigma_position_m": round(float(sigma_m), 4),
        "sigma_velocity_ms": round(float(sigma_v), 4),
    }


def summarise(tracks: pd.DataFrame) -> pd.DataFrame:
    """Per-track totals. Distance integrates the fitted speed, not raw steps."""
    df = tracks.dropna(subset=["speed_ms"]).sort_values(["track_id", "frame"])
    if df.empty:
        return pd.DataFrame(columns=["team", "track_id", "minutes", "distance_m",
                                     "top_speed_ms", "sprints"])
    dt = df.groupby("track_id")["t_s"].diff().fillna(0.0)
    df = df.assign(step_m=df["speed_ms"] * dt)
    return (
        df.groupby(["team", "track_id"], dropna=False)
        .agg(
            minutes=("t_s", lambda s: (s.max() - s.min()) / 60.0),
            distance_m=("step_m", "sum"),
            top_speed_ms=("speed_ms", "max"),
            sprints=("speed_ms", lambda s: int((s > 7.0).sum())),
        )
        .reset_index()
    )
