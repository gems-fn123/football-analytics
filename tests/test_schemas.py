import pandas as pd
import pytest

from footy.schemas import TRACKS_M, validate


def test_validate_accepts_complete_frame(tracks_m):
    out = validate(tracks_m, TRACKS_M, "tracks")
    assert out["x_m"].dtype == "float32"


def test_validate_rejects_missing_columns():
    df = pd.DataFrame({"frame": [1]})
    with pytest.raises(ValueError, match="missing columns"):
        validate(df, TRACKS_M, "tracks")
