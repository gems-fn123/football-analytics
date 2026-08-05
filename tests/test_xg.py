import numpy as np

from footy.analytics.xg import fit_base, predict, shot_features


def test_shot_features_shape(shots):
    feats = shot_features(shots)
    assert list(feats.columns) == ["distance_m", "angle_rad", "inverse_distance"]
    assert (feats["angle_rad"] >= 0).all()


def test_closer_shots_get_higher_xg(shots):
    model = fit_base(shots)
    import pandas as pd

    near = pd.DataFrame({"start_x": [100.0], "start_y": [34.0]})
    far = pd.DataFrame({"start_x": [72.0], "start_y": [12.0]})
    assert predict(model, near)[0] > predict(model, far)[0]


def test_xg_in_unit_interval(shots):
    model = fit_base(shots)
    p = predict(model, shots)
    assert np.all((p >= 0) & (p <= 1))
