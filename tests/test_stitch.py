import numpy as np

from footy.reid.stitch import stitch_tracks


def unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


def test_matching_fragments_merge():
    tracks = {
        1: {"start": 0, "end": 50, "embedding": unit([1, 0.1, 0])},
        2: {"start": 80, "end": 150, "embedding": unit([1, 0.12, 0])},
        3: {"start": 0, "end": 150, "embedding": unit([0, 1, 0])},
    }
    m = stitch_tracks(tracks)
    assert m[2] == m[1]
    assert m[3] != m[1]


def test_overlapping_tracks_never_merge():
    """Two tracks alive at the same time are two players, no matter the colours."""
    tracks = {
        1: {"start": 0, "end": 100, "embedding": unit([1, 0, 0])},
        2: {"start": 50, "end": 150, "embedding": unit([1, 0, 0])},
    }
    m = stitch_tracks(tracks)
    assert m[1] != m[2]


def test_ambiguous_candidates_abstain():
    """Two lookalike successors: merging either could be wrong, so merge neither."""
    tracks = {
        1: {"start": 0, "end": 50, "embedding": unit([1, 0, 0])},
        2: {"start": 80, "end": 150, "embedding": unit([1, 0.02, 0])},
        3: {"start": 82, "end": 149, "embedding": unit([1, 0.03, 0])},
    }
    m = stitch_tracks(tracks, margin=0.08)
    assert m[2] == 2 and m[3] == 3


def test_gap_beyond_limit_does_not_merge():
    tracks = {
        1: {"start": 0, "end": 50, "embedding": unit([1, 0, 0])},
        2: {"start": 1000, "end": 1100, "embedding": unit([1, 0, 0])},
    }
    m = stitch_tracks(tracks, max_gap_frames=250)
    assert m[2] == 2


def test_chain_merges_stay_time_consistent():
    e = unit([1, 0.1, 0.05])
    tracks = {
        1: {"start": 0, "end": 50, "embedding": e},
        2: {"start": 60, "end": 120, "embedding": e},
        3: {"start": 130, "end": 200, "embedding": e},
    }
    m = stitch_tracks(tracks)
    assert m[1] == m[2] == m[3]
