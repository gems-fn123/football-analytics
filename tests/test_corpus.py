import pandas as pd
import pytest

from footy.corpus.ingest.wikipedia import parse_league_table, parse_results_grid, strip_markers
from footy.corpus.provenance import PROVENANCE, stamp
from footy.corpus.store import CorpusStore

LEAGUE_HTML = """
<table class="wikitable">
<tr><th>Pos</th><th>Team</th><th>Pld</th><th>W</th><th>D</th><th>L</th>
<th>GF</th><th>GA</th><th>GD</th><th>Pts</th></tr>
<tr><td>1</td><td>Bali United (C)</td><td>34</td><td>20</td><td>8</td><td>6</td>
<td>60</td><td>32</td><td>+28</td><td>68</td></tr>
<tr><td>2</td><td>Persib Bandung</td><td>34</td><td>19</td><td>9</td><td>6</td>
<td>55</td><td>30</td><td>+25</td><td>66</td></tr>
<tr><td>18</td><td>Persiraja Banda Aceh (R)[a]</td><td>34</td><td>3</td><td>8</td><td>23</td>
<td>21</td><td>62</td><td>-41</td><td>17</td></tr>
</table>
"""

GRID_HTML = """
<table class="wikitable">
<tr><th>Home \\ Away</th><th>BAL</th><th>PSB</th><th>PSJ</th></tr>
<tr><td>Bali United</td><td>—</td><td>2–1</td><td>0–0</td></tr>
<tr><td>Persib Bandung</td><td>1–3</td><td>—</td><td>a</td></tr>
<tr><td>Persija Jakarta</td><td></td><td>2–0</td><td>—</td></tr>
</table>
"""


def make_stamped(df: pd.DataFrame) -> pd.DataFrame:
    return stamp(
        df,
        source="testsrc",
        source_url="https://example.org/x",
        licence_tag="CC0",
        ingestor_version="0.0.1",
    )


def test_strip_markers_removes_status_and_footnotes():
    assert strip_markers("Bali United (C)") == "Bali United"
    assert strip_markers("Persiraja Banda Aceh (R)[a]") == "Persiraja Banda Aceh"
    assert strip_markers("Persib Bandung") == "Persib Bandung"


def test_parse_league_table_extracts_aggregates():
    df = parse_league_table(LEAGUE_HTML)
    assert len(df) == 3
    top = df.iloc[0]
    assert top["club_name"] == "Bali United"
    assert top["club_raw"] == "Bali United (C)"  # verbatim survives for ER
    assert top["points"] == 68 and top["played"] == 34


def test_parse_results_grid_reads_scores_and_skips_unplayed():
    df = parse_results_grid(GRID_HTML)
    # 4 real scores; the diagonal, empty, and annotation cells are skipped.
    assert len(df) == 4
    row = df[(df["home_name"] == "Bali United") & (df["away_raw"] == "PSB")].iloc[0]
    assert (row["home_goals"], row["away_goals"]) == (2, 1)


GROUPED_HTML = """
<h2>First round<span>[edit]</span></h2>
<h3>West region</h3>
<table class="wikitable">
<tr><th>Pos</th><th>Team</th><th>Pld</th><th>Pts</th></tr>
<tr><td>1</td><td>PSMS</td><td>10</td><td>22</td></tr>
</table>
<h3>East region</h3>
<table class="wikitable">
<tr><th>Pos</th><th>Team</th><th>Pld</th><th>Pts</th></tr>
<tr><td>1</td><td>Kalteng Putra</td><td>10</td><td>20</td></tr>
</table>
<h2>Unrelated</h2>
<table class="wikitable"><tr><th>Name</th><th>Kit</th></tr><tr><td>x</td><td>y</td></tr></table>
"""


def test_group_labels_come_from_heading_path():
    df = parse_league_table(GROUPED_HTML)
    assert len(df) == 2  # the kit table is not standings-shaped
    by_group = dict(zip(df["group_raw"], df["club_name"], strict=True))
    assert by_group["First round / West region"] == "PSMS"
    assert by_group["First round / East region"] == "Kalteng Putra"


def test_store_roundtrip_and_partition_filters(tmp_path):
    store = CorpusStore(tmp_path)
    df = make_stamped(pd.DataFrame({"club_raw": ["Persib"], "points": [66]}))
    store.write("club_seasons", df, source="testsrc", competition="liga1", season="2024-25")
    got = store.read("club_seasons", competition="liga1")
    assert len(got) == 1 and got.iloc[0]["points"] == 66
    assert store.read("club_seasons", season="2017").empty


def test_store_rejects_rows_without_provenance(tmp_path):
    store = CorpusStore(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        store.write(
            "club_seasons",
            pd.DataFrame({"club_raw": ["Persib"]}),
            source="testsrc",
            competition="liga1",
            season="2024-25",
        )


def test_drop_source_quarantines_everything(tmp_path):
    store = CorpusStore(tmp_path)
    df = make_stamped(pd.DataFrame({"v": [1]}))
    store.write("club_seasons", df, source="badsrc", competition="liga1", season="2017")
    store.write("matches", df, source="badsrc", competition="liga1", season="2017")
    store.write("matches", df, source="goodsrc", competition="liga1", season="2017")
    assert store.drop_source("badsrc") == 2
    assert store.read("club_seasons").empty
    assert len(store.read("matches")) == 1  # the clean source survives


def test_provenance_columns_complete():
    df = make_stamped(pd.DataFrame({"v": [1]}))
    assert set(PROVENANCE) <= set(df.columns)
    assert df.iloc[0]["licence_tag"] == "CC0"


def test_fetcher_backs_off_on_429(tmp_path, monkeypatch):
    """429 responses are retried with backoff; success on a later try wins."""
    import requests as requests_mod

    from footy.corpus.ingest import base as base_mod

    calls = {"n": 0}

    class Resp:
        def __init__(self, status):
            self.status_code = status
            self.headers = {"Retry-After": "0"} if status == 429 else {"Content-Type": "text/html; charset=utf-8"}
            self.text = "payload"

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests_mod.HTTPError(f"{self.status_code}")

    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        return Resp(429 if calls["n"] < 3 else 200)

    monkeypatch.setattr(requests_mod, "get", fake_get)
    fetcher = base_mod.PoliteFetcher(
        "testsrc", raw_root=tmp_path, min_interval_s=0.0, respect_robots=False
    )
    assert fetcher.fetch("https://example.org/page", key="page") == "payload"
    assert calls["n"] == 3  # two 429s absorbed, third answered


def test_league_table_numbers_survive_footnote_markers():
    """Tied clubs' Pts render as '79[a]' on Wikipedia; they must parse as 79.

    Regression: to_numeric(coerce) NaN'd exactly the points-tied clubs,
    caught by cross-validating against footystats club totals."""
    html = """
    <html><body><h2>Standings</h2>
    <table class='wikitable'><tr><th>Pos</th><th>Teamvte</th><th>Pld</th>
    <th>W</th><th>D</th><th>L</th><th>GF</th><th>GA</th><th>Pts</th></tr>
    <tr><td>1</td><td>Persib (C)</td><td>34</td><td>24</td><td>7</td>
    <td>3</td><td>70</td><td>30</td><td>79[a]</td></tr>
    <tr><td>2</td><td>Borneo Samarinda</td><td>34</td><td>25</td><td>4</td>
    <td>5</td><td>74</td><td>31</td><td>79[a]</td></tr>
    </table></body></html>"""
    t = parse_league_table(html)
    assert t["points"].tolist() == [79, 79]
    assert t["position"].tolist() == [1, 2]


def test_cross_validate_club_seasons(tmp_path):
    import pandas as pd

    from footy.corpus.provenance import stamp
    from footy.corpus.validate import cross_validate_club_seasons

    store = CorpusStore(tmp_path)
    base = {
        "club_name": ["Persib Bandung", "Borneo FC"],
        "position": [1, 2],
        "played": [34, 34],
        "won": [24, 25],
        "drawn": [7, 4],
        "lost": [3, 5],
        "goals_for": [70, 74],
        "goals_against": [30, 31],
        "points": [79, 79],
    }
    a = pd.DataFrame(base)
    b = pd.DataFrame(base).assign(position=[2, 1], goals_for=[70, 73])  # one fact off
    for src, df in (("srcA", a), ("srcB", b)):
        df = stamp(df, source=src, source_url="u", licence_tag="t", ingestor_version="0")
        store.write("club_seasons", df, source=src, competition="liga1", season="2025-26")

    diffs = cross_validate_club_seasons(store, "srcA", "srcB", "liga1", "2025-26")
    kinds = diffs.set_index(["club_id", "field"])["kind"].to_dict()
    assert kinds[("borneo", "goals_for")] == "fact"
    assert kinds[("persib", "position")] == "ordering"
    assert len(diffs) == 3  # 1 fact + 2 ordering rows, nothing else
