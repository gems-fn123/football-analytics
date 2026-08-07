import pandas as pd
import pytest

from footy.corpus.entities import CLUBS, ClubResolver, normalise, resolve_clubs


@pytest.fixture
def r():
    return ClubResolver()


def test_normalise_strips_noise():
    assert normalise("Persib Bandung") == "persib bandung"
    assert normalise("Bali United FC") == "bali united"
    assert normalise("PS Barito Putera") == "barito putera"
    assert normalise("TIRA-Persikabo") == "tira persikabo"


# The six real variant pairs observed in the ingested corpus -----------------


def test_short_and_long_forms_resolve_to_same_club(r):
    assert r.resolve("Borneo") == r.resolve("Borneo Samarinda") == "borneo"
    assert r.resolve("Bhayangkara") == r.resolve("Bhayangkara Presisi") == "bhayangkara"
    assert r.resolve("Dewa United") == r.resolve("Dewa United Banten") == "dewa-united"
    assert r.resolve("PSPS") == r.resolve("PSPS Riau") == "psps-riau"
    assert r.resolve("Persita") == r.resolve("Persita Tangerang") == "persita"


def test_the_persik_trap_never_merges(r):
    """'Persik' is Kediri; 'Persik Kendal' is a different club. A prefix
    matcher merges them and corrupts every aggregate downstream."""
    assert r.resolve("Persik") == "persik-kediri"
    assert r.resolve("Persik Kendal") == "persik-kendal"


def test_renames_resolve_across_history(r):
    assert r.resolve("Persisam Putra") == "bali-united"
    assert r.resolve("Pusamania Borneo") == "borneo"
    assert r.resolve("Martapura Dewa United") == "dewa-united"


def test_one_letter_neighbours_do_not_cross(r):
    assert r.resolve("Persija") == "persija"
    assert r.resolve("Persijap") == "persijap"
    assert r.resolve("Persis") == "persis"
    assert r.resolve("PSIS") == "psis"


def test_unknown_and_youth_strings_abstain(r):
    assert r.resolve("Some Entirely New Club") is None
    assert r.resolve("Persib Bandung U20") is None  # youth side, never the seniors
    assert r.resolve("") is None
    assert "Some Entirely New Club" in r.unresolved()


def test_overrides_extend_but_cannot_invent_clubs():
    r = ClubResolver(overrides={"Maung Bandung": "persib"})
    assert r.resolve("Maung Bandung") == "persib"
    with pytest.raises(ValueError, match="not a registered club"):
        ClubResolver(overrides={"X": "no-such-club"})


def test_alias_uniqueness_is_enforced_at_import():
    seen = {}
    for cid, spec in CLUBS.items():
        for a in spec["aliases"]:
            assert a not in seen, f"alias {a} in both {seen.get(a)} and {cid}"
            seen[a] = cid


def test_resolve_clubs_adds_nullable_id_column(r):
    df = pd.DataFrame({"club_raw": ["Persib", "Persik Kendal", "Mystery FC"]})
    out = resolve_clubs(df, "club_raw", r)
    assert out["club_id"].tolist()[:2] == ["persib", "persik-kendal"]
    assert pd.isna(out["club_id"].iloc[2])
