"""Match analytics from pitch-normalised tracks.

Positions are recoverable from this footage; their *derivatives* are much harder,
because summing per-frame displacement turns measurement noise into distance. A
constant-velocity Kalman filter with RTS smoothing is used rather than a rolling mean:
it models measurement noise explicitly, so the velocity estimate is not simply a
low-pass of the noise. The residual noise floor is measured and reported rather than
hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pitch_model as pm  # noqa: E402

FPS = 25.0
SPRINT_MS = 7.0
HIGH_SPEED_MS = 5.5


# ------------------------------------------------------------------ smoothing

def rts_smooth(z: np.ndarray, dt: float, q: float = 0.55, r: float = 0.65):
    """Constant-velocity Kalman filter + RTS smoother on a 2-D track.

    q  process noise (m/s^2): how hard a footballer can change velocity.
    r  measurement noise (m): box-bottom jitter reprojected to the pitch. At 768x432
       and ~0.05 m/px a couple of pixels of box wobble is several tens of centimetres,
       which is why r is this large relative to the motion.
    Returns smoothed positions and velocities.
    """
    n = len(z)
    if n < 3:
        return z.copy(), np.zeros_like(z)
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]], float)
    Q = G @ (q**2 * np.eye(2)) @ G.T
    R = r**2 * np.eye(2)

    xf = np.zeros((n, 4))
    Pf = np.zeros((n, 4, 4))
    xp = np.zeros((n, 4))
    Pp = np.zeros((n, 4, 4))

    x = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
    P = np.diag([r**2, r**2, 4.0, 4.0])
    for k in range(n):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        y = z[k] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P
        xf[k], Pf[k] = x, P

    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs[:, :2], xs[:, 2:4]


def add_kinematics(tracks: pd.DataFrame) -> pd.DataFrame:
    out = []
    for tid, d in tracks.sort_values(["track_id", "frame"]).groupby("track_id"):
        d = d.copy()
        z = d[["x_m", "y_m"]].to_numpy(float)
        pos, vel = rts_smooth(z, 1.0 / FPS)
        d["x_s"], d["y_s"] = pos[:, 0], pos[:, 1]
        d["vx"], d["vy"] = vel[:, 0], vel[:, 1]
        d["speed_ms"] = np.hypot(vel[:, 0], vel[:, 1])
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else tracks.assign(speed_ms=np.nan)


def noise_floor(tracks: pd.DataFrame) -> dict:
    """Estimate the residual measurement noise, so distance can be error-barred.

    The filter's own innovation - the gap between raw measurement and smoothed
    position - is a direct read on how noisy the input is.
    """
    r = np.hypot(tracks["x_m"] - tracks["x_s"], tracks["y_m"] - tracks["y_s"])
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    resid = float(np.median(r)) if len(r) else float("nan")
    # A random walk of this size per frame would fabricate this much distance per minute
    fabricated_m_per_min = resid * np.sqrt(2) * FPS * 60 / 1000 * 1000
    return {
        "median_residual_m": round(resid, 3),
        "implied_noise_distance_m_per_min": round(float(fabricated_m_per_min), 1),
    }


# ------------------------------------------------------------------ identities

# Positional bands, not detected roles. There is deliberately no "GK": a goalkeeper is
# identified by kit and by standing in a goal, and at kickoff both keepers are off
# camera entirely. Calling the deepest visible track a goalkeeper would be an invention.
ROLE_ORDER = ["DEEP", "DEEP", "DEEP", "DEEP", "MID", "MID", "MID", "MID", "HIGH", "HIGH", "HIGH"]


def stitch_tracks(
    tracks: pd.DataFrame,
    max_gap_s: float = 2.0,
    max_dist_m: float = 7.0,
) -> pd.DataFrame:
    """Join track fragments that are the same player, in pitch space.

    ByteTrack fragments badly here - 87 ids for ~15 people on clip0 - because players
    occlude each other and leave frame. Stitching is done in *metres* rather than
    pixels, which is the whole point of normalising first: a gap of 7 m means the same
    thing wherever it happens on the pitch, whereas 7 pixels does not.

    A fragment is joined to its predecessor when the predecessor ends shortly before it
    begins, the two are close once the predecessor's last velocity is extrapolated over
    the gap, and both carry the same team label. Greedy, nearest-first.
    """
    tracks = tracks.copy()
    ends = []
    for tid, d in tracks.groupby("track_id"):
        d = d.sort_values("frame")
        ends.append(
            {
                "track_id": tid,
                "team": d["team"].dropna().iloc[0] if d["team"].notna().any() else None,
                "t0": float(d["t_s"].iloc[0]),
                "t1": float(d["t_s"].iloc[-1]),
                "x0": float(d["x_s"].iloc[0]),
                "y0": float(d["y_s"].iloc[0]),
                "x1": float(d["x_s"].iloc[-1]),
                "y1": float(d["y_s"].iloc[-1]),
                "vx": float(d["vx"].iloc[-1]) if "vx" in d else 0.0,
                "vy": float(d["vy"].iloc[-1]) if "vy" in d else 0.0,
                "n": len(d),
            }
        )
    info = pd.DataFrame(ends).sort_values("t0").reset_index(drop=True)

    parent = {t: t for t in info["track_id"]}

    def root(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    used_successor: set = set()
    for _, a in info.iterrows():
        best, best_d = None, max_dist_m
        for _, b in info.iterrows():
            if b["track_id"] == a["track_id"] or b["track_id"] in used_successor:
                continue
            gap = b["t0"] - a["t1"]
            if gap <= 0 or gap > max_gap_s:
                continue
            if a["team"] and b["team"] and a["team"] != b["team"]:
                continue
            if root(a["track_id"]) == root(b["track_id"]):
                continue
            px = a["x1"] + a["vx"] * gap
            py = a["y1"] + a["vy"] * gap
            d = float(np.hypot(b["x0"] - px, b["y0"] - py))
            if d < best_d:
                best, best_d = b["track_id"], d
        if best is not None:
            used_successor.add(best)
            parent[root(best)] = root(a["track_id"])

    tracks["player_id"] = tracks["track_id"].map(lambda t: root(t) if t in parent else t)
    return tracks


def assign_synthetic_identities(tracks: pd.DataFrame, min_seconds: float = 2.0) -> pd.DataFrame:
    """Give every persistent track a stable synthetic shirt number and role.

    Real numbers are unreadable at this resolution - a human cannot read them from
    these frames either - and SoccerNet reports jersey recognition as the weakest link
    in the whole game-state pipeline even at broadcast resolution. So identity is
    *constructed*: each track that survives long enough becomes a numbered synthetic
    player, ordered by mean position along the pitch so the numbering carries positional
    meaning (1 = deepest, 11 = most advanced) the way real squad numbers loosely do.

    These are labels for a track, not claims about a person. They are prefixed to make
    that impossible to miss.
    """
    tracks = tracks.copy()
    tracks["synth_id"] = pd.NA
    tracks["role"] = pd.NA
    key = "player_id" if "player_id" in tracks.columns else "track_id"

    # Attacking direction is decided once, from the two teams *relative* to each other.
    # Judging each side on its own ("is my centroid past halfway?") fails during a
    # sustained attack, when both teams are camped in the same half: the defending side
    # sits deeper in its own territory, so the team with the smaller mean x is the one
    # attacking toward the larger x, whatever the absolute numbers say.
    centroids = {
        t: float(tracks.loc[tracks["team"] == t, "x_s"].mean())
        for t in ("home", "away")
        if (tracks["team"] == t).any()
    }
    attack_right = {}
    if len(centroids) == 2:
        low = min(centroids, key=centroids.get)
        for t in centroids:
            attack_right[t] = t == low
    else:
        for t in centroids:
            attack_right[t] = centroids[t] < pm.CX

    for team in ("home", "away"):
        t = tracks[tracks["team"] == team]
        if not len(t):
            continue
        # Total observed time, summed across the fragments that were stitched together.
        life = t.groupby(key)["t_s"].agg(lambda s: s.max() - s.min())
        keep = life[life >= min_seconds].index
        # A side has eleven players. Keep the eleven best-observed identities and leave
        # the rest unnamed rather than inventing a 24-man team on the pitch.
        if len(keep) > 11:
            keep = life.loc[keep].sort_values(ascending=False).head(11).index
        if not len(keep):
            continue
        # Attacking direction: mean x of the team decides which way "forward" is, so
        # numbering is consistent regardless of which end they attack.
        mean_x = t[t[key].isin(keep)].groupby(key)["x_s"].mean().sort_values()
        order = mean_x.index.tolist()
        # 1 = deepest relative to this team's own goal, 11 = most advanced.
        if not attack_right.get(team, True):
            order = order[::-1]
        for i, tid in enumerate(order):
            num = i + 1
            role = ROLE_ORDER[i] if i < len(ROLE_ORDER) else "SUB"
            prefix = "H" if team == "home" else "A"
            tracks.loc[tracks[key] == tid, "synth_id"] = f"{prefix}{num:02d}"
            tracks.loc[tracks[key] == tid, "role"] = role
    return tracks


# ------------------------------------------------------------------ team metrics

def player_table(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (team, sid), d in tracks.dropna(subset=["synth_id"]).groupby(["team", "synth_id"]):
        d = d.sort_values("frame")
        secs = float(d["t_s"].max() - d["t_s"].min())
        if secs <= 0:
            continue
        step = np.hypot(d["x_s"].diff(), d["y_s"].diff())
        dist = float(np.nansum(step))
        rows.append(
            {
                "team": team,
                "player": sid,
                "role": d["role"].iloc[0],
                "minutes_on_camera": round(secs / 60, 2),
                "distance_m": round(dist, 1),
                "m_per_min": round(dist / (secs / 60), 1),
                "top_speed_ms": round(float(d["speed_ms"].quantile(0.98)), 2),
                "high_speed_s": round(float((d["speed_ms"] > HIGH_SPEED_MS).sum() / FPS), 1),
                "sprints": int(_count_bursts(d["speed_ms"].to_numpy(), SPRINT_MS)),
                "mean_x_m": round(float(d["x_s"].mean()), 1),
                "mean_y_m": round(float(d["y_s"].mean()), 1),
                "touches_est": int(d.get("is_carrier", pd.Series(dtype=bool)).sum())
                if "is_carrier" in d
                else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["team", "player"]).reset_index(drop=True)


def _count_bursts(speed: np.ndarray, thresh: float, min_frames: int = 5) -> int:
    above = speed > thresh
    n, run = 0, 0
    for a in above:
        if a:
            run += 1
        else:
            if run >= min_frames:
                n += 1
            run = 0
    return n + (1 if run >= min_frames else 0)


def team_shape(tracks: pd.DataFrame) -> pd.DataFrame:
    """Per-frame team shape: centroid, width, depth, defensive line height."""
    rows = []
    for (team, frame), d in tracks[tracks["team"].isin(["home", "away"])].groupby(
        ["team", "frame"]
    ):
        if len(d) < 4:
            continue
        rows.append(
            {
                "team": team,
                "frame": frame,
                "t_s": d["t_s"].iloc[0],
                "n": len(d),
                "cx": d["x_s"].mean(),
                "cy": d["y_s"].mean(),
                "width_m": d["y_s"].max() - d["y_s"].min(),
                "depth_m": d["x_s"].max() - d["x_s"].min(),
            }
        )
    return pd.DataFrame(rows)


def possession(tracks: pd.DataFrame, ball: pd.DataFrame, max_dist_m: float = 3.0) -> pd.DataFrame:
    """Nearest-player possession proxy.

    Not an event-detected possession: the ball's ground position is approximate
    whenever it is airborne, so this is 'who is closest to the ball', held to a
    distance gate. Frames where nobody is within the gate are unattributed rather than
    assigned to the least-far player.
    """
    if not len(ball):
        return pd.DataFrame(columns=["frame", "t_s", "team", "player", "dist_m"])
    b = ball.dropna(subset=["x_m", "y_m"]).set_index("frame")
    rows = []
    for frame, d in tracks[tracks["team"].isin(["home", "away"])].groupby("frame"):
        if frame not in b.index:
            continue
        bx, by = float(b.loc[frame, "x_m"]), float(b.loc[frame, "y_m"])
        dist = np.hypot(d["x_s"] - bx, d["y_s"] - by)
        i = dist.idxmin()
        if dist.loc[i] <= max_dist_m:
            rows.append(
                {
                    "frame": frame,
                    "t_s": d.loc[i, "t_s"],
                    "team": d.loc[i, "team"],
                    "player": d.loc[i, "synth_id"],
                    "dist_m": round(float(dist.loc[i]), 2),
                    "x_m": float(d.loc[i, "x_s"]),
                    "y_m": float(d.loc[i, "y_s"]),
                }
            )
    return pd.DataFrame(rows)
