"""Stage 3: split tracks into home, away, referee.

Plain version: crop each player's torso, look at the shirt colour, decide which
side they are on. The match config supplies the kit colours.

No roster needed. This is why the pipeline works on a league with no public data.

Colour statistic: median Lab chroma (a, b) of grass-masked torso crops, one point
per track. Chroma-only makes it robust to the shadow bands and exposure shifts that
wreck RGB or HSV-value clustering.

Assignment is relative, not absolute. Real crops are small and motion-blurred, so
their colours come out muted: comparing them against saturated config hexes with a
distance cutoff throws most tracks away (and neutral references like a white kit
vacuum up everything). Instead the tracks are clustered on their empirical colours,
and only the *cluster centroids* are compared against the config kit colours - a
relative ordering that survives muting. Goalkeepers and referees fall outside the
two big field-player clusters and are matched against the gk/referee config colours;
tracks that match nothing stay null. A null team is honest, a guessed one corrupts
possession and every team metric downstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from footy.stages.base import Stage, StageResult, upstream_table

# Grass in HSV: hue band around green with real saturation. Used to mask pitch
# pixels out of the torso crop before taking the median.
GRASS_HUE = (35, 90)
GRASS_MIN_SAT = 60


def hex_to_ab(hex_colour: str) -> np.ndarray:
    """Kit colour hex -> Lab chroma (a, b), matching the crop statistic."""
    import cv2

    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    patch = np.array([[[b, g, r]]], dtype=np.uint8)  # BGR for cv2
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab)
    return lab[0, 0, 1:3].astype(np.float64)


def torso_chroma(
    frame_bgr: np.ndarray, box: tuple[float, float, float, float]
) -> np.ndarray | None:
    """Median (a, b) of the torso region with grass pixels masked out."""
    import cv2

    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    if h < 8 or w < 4:
        return None
    # Torso: below the head, above the shorts, central to avoid background bleed.
    ty1, ty2 = int(y1 + 0.15 * h), int(y1 + 0.50 * h)
    tx1, tx2 = int(x1 + 0.25 * w), int(x1 + 0.75 * w)
    ty1, tx1 = max(ty1, 0), max(tx1, 0)
    ty2, tx2 = min(ty2, frame_bgr.shape[0]), min(tx2, frame_bgr.shape[1])
    crop = frame_bgr[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (
        (hsv[..., 0] >= GRASS_HUE[0])
        & (hsv[..., 0] <= GRASS_HUE[1])
        & (hsv[..., 1] >= GRASS_MIN_SAT)
    )
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    kit = lab[~grass]
    if len(kit) < 12:  # crop is nearly all grass, unusable
        return None
    return np.median(kit[:, 1:3].astype(np.float64), axis=0)


class TeamAssigner(Stage):
    name = "team"

    def setup(self) -> None:
        self.method = self.cfg.get("method", "kit_colour")
        if self.method == "siglip_umap_kmeans":
            raise NotImplementedError(
                "siglip_umap_kmeans needs a SigLIP encoder pass; use method: kit_colour"
            )
        if self.method != "kit_colour":
            raise ValueError(f"unknown team assignment method: {self.method!r}")
        self.sample_stride = max(1, int(self.cfg.get("sample_stride", 5)))
        self.min_crops = int(self.cfg.get("min_crops", 3))
        # Boxes shorter than this give unusable torso crops; skip them.
        self.min_box_h = float(self.cfg.get("min_box_h", 16.0))
        # Lab chroma units, all relative to *empirical* centroids or between
        # empirical values and config hexes where only the ordering matters.
        self.outlier_dist = float(self.cfg.get("outlier_dist", 25.0))
        self.special_match_dist = float(self.cfg.get("special_match_dist", 40.0))
        self.log.info("team assignment method=%s stride=%d", self.method, self.sample_stride)

    def _collect_track_chroma(self, source: Any, tracks: pd.DataFrame) -> dict[int, np.ndarray]:
        """One decode pass over the video, median chroma per track."""
        tracked = tracks[
            (tracks["track_id"] >= 0)
            & (tracks["cls"] != "ball")
            & ((tracks["y2"] - tracks["y1"]) >= self.min_box_h)
        ]
        wanted = tracked[tracked["frame"] % self.sample_stride == 0]
        by_frame = {int(f): g for f, g in wanted.groupby("frame")}
        if not by_frame:
            return {}

        samples: dict[int, list[np.ndarray]] = {}
        last_needed = max(by_frame)
        for frame_idx, image in source:
            group = by_frame.get(frame_idx)
            if group is not None:
                for row in group.itertuples():
                    ab = torso_chroma(image, (row.x1, row.y1, row.x2, row.y2))
                    if ab is not None:
                        samples.setdefault(int(row.track_id), []).append(ab)
            if frame_idx >= last_needed:
                break

        return {
            tid: np.median(np.stack(abs_), axis=0)
            for tid, abs_ in samples.items()
            if len(abs_) >= self.min_crops
        }

    @staticmethod
    def _special_colours(match: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
        """(team value, chroma) for goalkeeper and referee reference colours."""
        specials: list[tuple[str, np.ndarray]] = []
        for side in ("home", "away"):
            gk = (match.get(side) or {}).get("gk_kit_colour")
            if gk:
                specials.append((side, hex_to_ab(gk)))
        if match.get("referee_colour"):
            specials.append(("referee", hex_to_ab(match["referee_colour"])))
        return specials

    def _label_clusters(
        self, centroids: np.ndarray, cluster_a: int, cluster_b: int, match: dict[str, Any]
    ) -> dict[int, str]:
        """Name the two team clusters home/away using whatever kit colours exist.

        Only the relative ordering of centroid-to-hex distances is used, so muted
        video colours against saturated config hexes still map correctly. With one
        kit colour known, its nearest centroid claims it and the other cluster gets
        the remaining label. With none, size order decides, with a warning.
        """
        home_hex = (match.get("home") or {}).get("kit_colour")
        away_hex = (match.get("away") or {}).get("kit_colour")

        if home_hex and away_hex:
            h, a = hex_to_ab(home_hex), hex_to_ab(away_hex)
            straight = np.linalg.norm(centroids[cluster_a] - h) + np.linalg.norm(
                centroids[cluster_b] - a
            )
            swapped = np.linalg.norm(centroids[cluster_a] - a) + np.linalg.norm(
                centroids[cluster_b] - h
            )
            if straight <= swapped:
                return {cluster_a: "home", cluster_b: "away"}
            return {cluster_a: "away", cluster_b: "home"}

        if home_hex or away_hex:
            known_label = "home" if home_hex else "away"
            other_label = "away" if home_hex else "home"
            ref = hex_to_ab(home_hex or away_hex)
            d_a = float(np.linalg.norm(centroids[cluster_a] - ref))
            d_b = float(np.linalg.norm(centroids[cluster_b] - ref))
            self.log.info(
                "only the %s kit colour is configured; matching it by nearest cluster",
                known_label,
            )
            if d_a <= d_b:
                return {cluster_a: known_label, cluster_b: other_label}
            return {cluster_a: other_label, cluster_b: known_label}

        self.log.warning(
            "no kit colours in the match config; home/away is an arbitrary labelling. "
            "Add kit_colour under home: and away: for a deterministic mapping."
        )
        return {cluster_a: "home", cluster_b: "away"}

    def _fit_team_centroids(self, X: np.ndarray) -> np.ndarray:
        """Outlier-peeling k-means: two clean field-player centroids.

        A referee or goalkeeper in the sample drags a plain k-means centroid toward
        itself, which can push genuine kit tracks past the outlier bound. So: fit
        k=2, and while the worst-fitting point is beyond outlier_dist, remove it and
        refit. Contaminants are few, so this converges in a handful of rounds.
        """
        from sklearn.cluster import KMeans

        keep = np.arange(len(X))
        while len(keep) > 2:
            km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X[keep])
            dists = np.linalg.norm(X[keep] - km.cluster_centers_[km.labels_], axis=1)
            worst = int(np.argmax(dists))
            if dists[worst] <= self.outlier_dist:
                # Order centroids by final cluster size, biggest first.
                sizes = [int((km.labels_ == c).sum()) for c in (0, 1)]
                order = [0, 1] if sizes[0] >= sizes[1] else [1, 0]
                return km.cluster_centers_[order]
            keep = np.delete(keep, worst)
        return X[keep]  # two (or fewer) points left: they are the centroids

    def _assign(self, track_ab: dict[int, np.ndarray], match: dict[str, Any]) -> dict[int, str]:
        tids = sorted(track_ab)
        X = np.stack([track_ab[t] for t in tids])
        centroids = self._fit_team_centroids(X)
        cluster_team = self._label_clusters(centroids, 0, 1, match)
        specials = self._special_colours(match)

        team_of: dict[int, str] = {}
        for i, tid in enumerate(tids):
            dists = np.linalg.norm(centroids - X[i], axis=1)
            nearest = int(np.argmin(dists))
            if float(dists[nearest]) <= self.outlier_dist:
                team_of[tid] = cluster_team[nearest]
                continue
            # Outlier: goalkeeper/referee if the config names a colour close
            # enough; otherwise honest null.
            best_label, best_d = None, self.special_match_dist
            for team_value, ab in specials:
                d = float(np.linalg.norm(X[i] - ab))
                if d < best_d:
                    best_label, best_d = team_value, d
            if best_label is not None:
                team_of[tid] = best_label
        return team_of

    def run(self, ctx: dict[str, Any]) -> StageResult:
        tracks = upstream_table(ctx, "track", stage=self.name)
        match = ctx["config"].match or {}

        track_ab = self._collect_track_chroma(ctx["video"], tracks)
        if len(track_ab) >= 2:
            team_of = self._assign(track_ab, match)
        else:
            if not track_ab:
                self.log.warning("no tracks with enough crops; team stays null everywhere")
            else:
                self.log.warning("only one track with enough crops; cannot cluster")
            team_of = {}

        team = tracks["track_id"].map(team_of)
        team = team.where(tracks["cls"] != "ball", "ball")
        df = tracks.assign(team=team.astype("string"))

        counts = pd.Series(list(team_of.values())).value_counts().to_dict() if team_of else {}
        stats = {
            "n_tracks_sampled": len(track_ab),
            "teams": counts,
            "n_unassigned": len(track_ab) - len(team_of),
        }
        return StageResult(self.name, df, stats=stats)
