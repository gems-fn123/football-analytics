"""Expected goals.

The geometry transfers across leagues. The conversion rate does not. A model fitted
on elite European shots will overstate xG in a weaker or slower league, because the
same shot is converted less often.

So: fit the shape on open data, then recalibrate the intercept on local shots.
Roughly 250 local shots with known outcomes is enough for a usable recalibration.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

GOAL_X_M = 105.0
GOAL_Y_M = 34.0
GOAL_WIDTH_M = 7.32


def shot_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Distance and angle to goal. The two features that carry across leagues."""
    dx = GOAL_X_M - shots["start_x"]
    dy = GOAL_Y_M - shots["start_y"]
    dist = np.hypot(dx, dy)
    half = GOAL_WIDTH_M / 2
    angle = np.arctan2(
        GOAL_WIDTH_M * dx,
        dx**2 + dy**2 - half**2,
    )
    angle = np.where(angle < 0, angle + np.pi, angle)
    return pd.DataFrame({
        "distance_m": dist,
        "angle_rad": angle,
        "inverse_distance": 1 / dist.clip(lower=1.0),
    })


def fit_base(shots: pd.DataFrame, target: str = "is_goal"):
    """Fit on open data (StatsBomb or Wyscout). Returns a fitted estimator."""
    from sklearn.linear_model import LogisticRegression

    X = shot_features(shots)
    y = shots[target].astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def fit_local(shots_path: str | Path, out_path: str | Path, base_model_path: str | None = None):
    """Recalibrate on local shots. Refits the intercept, keeps the shape.

    TODO: implement Platt scaling against the base model rather than a full refit
    when the local sample is under ~250 shots.
    """
    shots = (
        pd.read_parquet(shots_path)
        if str(shots_path).endswith(".parquet")
        else pd.read_csv(shots_path)
    )
    if len(shots) < 100:
        raise ValueError(f"only {len(shots)} local shots, need ~250 for a stable recalibration")
    model = fit_base(shots)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(model, fh)
    return model


def predict(model, shots: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(shot_features(shots))[:, 1]
