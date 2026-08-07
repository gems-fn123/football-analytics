import pandas as pd
import pytest

from footy.corpus.ingest.footystats import FootyStats, season_key

MATCHES_FIXTURE = [
    {
        "id": 101,
        "status": "complete",
        "date_unix": 1755907200,
        "home_name": "Persib Bandung",
        "away_name": "PSM Makassar",
        "homeGoalCount": 2,
        "awayGoalCount": 1,
        "homeID": 7,
        "awayID": 9,
    },
    {"id": 102, "status": "incomplete", "home_name": "A", "away_name": "B"},
    {"id": 103, "status": "suspended", "home_name": "C", "away_name": "D"},
]

PLAYERS_FIXTURE = [
    {
        "id": 555,
        "full_name": "Marc Klok",
        "known_as": "Marc Klok",
        "position": "Midfielder",
        "club_team_id": 7,
        "appearances_overall": 30,
        "minutes_played_overall": 2610,
        "goals_overall": 6,
        "assists_overall": 4,
        "yellow_cards_overall": 5,
        "red_cards_overall": 0,
        "clean_sheets_overall": 0,
    },
    {
        # Mononym with no club mapping: club_raw must be empty, not guessed.
        "id": 556,
        "full_name": "Beckham",
        "known_as": "Beckham Putra",
        "position": "Forward",
        "club_team_id": 99,
    },
]


def test_season_key_formats():
    assert season_key(2017) == "2017"
    assert season_key(20212022) == "2021-22"
    assert season_key("20242025") == "2024-25"


def test_normalise_matches_keeps_only_complete():
    df = FootyStats.normalise_matches(MATCHES_FIXTURE)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["home_raw"] == "Persib Bandung"
    assert (row["home_goals"], row["away_goals"]) == (2, 1)
    assert row["date"] == "2025-08-23"


def test_normalise_players_maps_club_and_abstains_when_unknown():
    df = FootyStats.normalise_players(PLAYERS_FIXTURE, club_names={7: "Persib Bandung"})
    klok = df[df["player_raw"] == "Marc Klok"].iloc[0]
    assert klok["club_raw"] == "Persib Bandung"
    assert klok["minutes"] == 2610
    beckham = df[df["player_raw"] == "Beckham"].iloc[0]
    assert beckham["club_raw"] == ""  # unknown club id 99 -> empty, never guessed
    assert pd.isna(beckham["appearances"]) or beckham["appearances"] is None


def test_api_key_error_when_absent(monkeypatch, tmp_path):
    from footy.corpus.ingest import footystats as fs

    monkeypatch.delenv("FOOTYSTATS_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    with pytest.raises(RuntimeError, match="FOOTYSTATS_API_KEY"):
        fs.api_key()
