from conftest import (
    KIT_BLUE_BGR,
    KIT_BLUE_HEX,
    KIT_RED_BGR,
    KIT_RED_HEX,
    KIT_YELLOW_BGR,
    KIT_YELLOW_HEX,
    FakeVideo,
    synthetic_tracks_px,
)

from footy.config import Config
from footy.stages.base import StageResult
from footy.stages.team import TeamAssigner, hex_to_ab, torso_chroma

BOXES = {
    0: (KIT_RED_BGR, 50, 100, 80, 160),
    1: (KIT_RED_BGR, 150, 100, 180, 160),
    2: (KIT_BLUE_BGR, 350, 100, 380, 160),
    3: (KIT_BLUE_BGR, 450, 100, 480, 160),
    4: (KIT_YELLOW_BGR, 550, 100, 580, 160),  # referee
}

MATCH = {
    "home": {"kit_colour": KIT_RED_HEX},
    "away": {"kit_colour": KIT_BLUE_HEX},
    "referee_colour": KIT_YELLOW_HEX,
}


def run_team(match: dict, boxes: dict = BOXES) -> StageResult:
    stage = TeamAssigner({"method": "kit_colour", "sample_stride": 1, "min_crops": 3})
    stage.setup()
    ctx = {
        "video": FakeVideo(boxes),
        "track": StageResult("track", synthetic_tracks_px(boxes)),
        "config": Config(raw={}, match=match),
    }
    return stage.run(ctx)


def test_clusters_map_to_config_kit_colours():
    df = run_team(MATCH).table
    by_track = df[df["cls"] == "player"].groupby("track_id")["team"].first()
    assert by_track[0] == "home" and by_track[1] == "home"
    assert by_track[2] == "away" and by_track[3] == "away"


def test_outlier_matches_referee_colour():
    df = run_team(MATCH).table
    assert df[df["track_id"] == 4]["team"].iloc[0] == "referee"


def test_unmatched_outlier_stays_null_not_guessed():
    match = {k: v for k, v in MATCH.items() if k != "referee_colour"}
    df = run_team(match).table
    assert df[df["track_id"] == 4]["team"].isna().all()


def test_ball_rows_are_labelled_ball():
    df = run_team(MATCH).table
    assert (df.loc[df["cls"] == "ball", "team"] == "ball").all()


def test_no_kit_colours_still_splits_two_teams():
    df = run_team({}).table
    players = df[(df["cls"] == "player") & (df["track_id"] < 4)]
    teams = players.groupby("track_id")["team"].first()
    assert set(teams) == {"home", "away"}
    # Same-colour tracks land on the same side even without a config.
    assert teams[0] == teams[1] and teams[2] == teams[3] and teams[0] != teams[2]
    # The referee is an outlier with no reference colour to name them: null, not
    # absorbed into a team.
    assert df[df["track_id"] == 4]["team"].isna().all()


def test_single_kit_colour_still_labels_deterministically():
    """Partial config: only home's kit is known. Its nearest cluster claims home."""
    df = run_team({"home": {"kit_colour": KIT_RED_HEX}}).table
    by_track = df[df["cls"] == "player"].groupby("track_id")["team"].first()
    assert by_track[0] == "home" and by_track[1] == "home"
    assert by_track[2] == "away" and by_track[3] == "away"


def test_hex_to_ab_separates_the_kits():
    import numpy as np

    red, blue = hex_to_ab(KIT_RED_HEX), hex_to_ab(KIT_BLUE_HEX)
    assert np.linalg.norm(red - blue) > 40


def test_torso_chroma_rejects_grass_only_crops():
    import numpy as np

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = (30, 160, 30)  # all grass
    assert torso_chroma(frame, (10, 10, 60, 90)) is None
