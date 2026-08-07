import numpy as np
import pandas as pd
import pytest

from footy.corpus.percentiles import MIN_POPULATION, percentile


def corpus(n=100, competition="liga1", position="Midfielder", season="2024-25"):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "player_raw": [f"P{i}" for i in range(n)],
            "position": position,
            "competition": competition,
            "season": season,
            "minutes": rng.integers(500, 3000, n),
            "goals": np.arange(n),  # 0..n-1: exact ranks known
        }
    )


def test_percentile_is_empirical_rank():
    r = percentile(corpus(), "goals", value=90, competition="liga1")
    assert r is not None
    assert r.percentile == pytest.approx(90.5)  # 90 below, 1 tie of 100
    assert r.n == 100 and r.median == pytest.approx(49.5)


def test_small_population_abstains():
    r = percentile(corpus(n=MIN_POPULATION - 1), "goals", value=5)
    assert r is None


def test_filters_shrink_population_honestly():
    df = pd.concat([corpus(), corpus(competition="liga2")], ignore_index=True)
    r = percentile(df, "goals", value=50, competition="liga1")
    assert r.n == 100  # liga2 rows excluded, and the result says so


def test_minutes_floor_excludes_cameos():
    df = corpus()
    df.loc[:49, "minutes"] = 10  # 50 players below the floor
    r = percentile(df, "goals", value=75, competition="liga1")
    assert r.n == 50


def test_per90_is_derived_from_own_minutes():
    df = corpus()
    df["minutes"] = 900  # everyone: goals per90 = goals / 10
    r = percentile(df, "goals_per90", value=5.0, competition="liga1")
    assert r is not None
    # value 5.0 per90 = 50 goals: 50 strictly below, one tie
    assert r.percentile == pytest.approx(50.5)


def test_unknown_metric_raises():
    with pytest.raises(KeyError, match="not derivable"):
        percentile(corpus(), "xg_chain", value=1.0)


def test_sentence_is_presentable():
    r = percentile(corpus(), "goals", value=90, competition="liga1", position="Midfielder")
    s = r.sentence()
    assert "90th percentile" in s and "Midfielder" in s and "100" in s
