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
