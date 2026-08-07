"""footystats.org scraper: parsers against fixture markup + ingest wiring.

Fixtures are trimmed copies of the real page structures (verified live on
2026-08-07): the full-league-table, the schema.org squad rows, the
player_season_row career blocks, and the per-competition stat tables.
"""

import pandas as pd
import pytest

from footy.corpus.ingest.footystats_web import (
    FootyStatsWeb,
    league_season,
    parse_career,
    parse_current_season,
    parse_league_table,
    parse_squad,
    season_key_web,
)
from footy.corpus.store import CorpusStore

LEAGUE_HTML = """
<html><head><title>Indonesia Liga 1 2025/26 Stats | FootyStats</title></head><body>
<table class='full-league-table'><thead><tr><th>Pos</th><th>Team</th></tr></thead><tbody>
<tr>
 <td class='position bold'><span>1</span></td>
 <td class='team'><a href='/clubs/borneo-fc-1536' data-team-id='1536' data-comp-id='15283'>Borneo FC</a></td>
 <td class='mp'>34</td><td class='win'>25</td><td class='draw'>4</td><td class='loss'>5</td>
 <td class='gf'>74</td><td class='ga'>31</td><td class='points bold'>79</td>
 <td class='over05'><a class='bold' href='/premium'><i class='fas fa-lock'></i></a></td>
</tr>
<tr>
 <td class='position'><span>2</span></td>
 <td class='team'><a href='/clubs/persib-1532' data-team-id='1532' data-comp-id='15283'>Persib</a></td>
 <td class='mp'>34</td><td class='win'>24</td><td class='draw'>7</td><td class='loss'>3</td>
 <td class='gf'>70</td><td class='ga'>30</td><td class='points'>79</td>
 <td class='over05'>76%</td>
</tr>
</tbody></table></body></html>
"""

# Real pages render every squad row twice (desktop + mobile) - so does this.
SQUAD_HTML = """
<html><body>
<div itemprop='member' itemscope itemtype='http://schema.org/OrganizationRole'>
 <p><a href='/players/spain/koldo-obieta-alberdi'>Koldo Obieta Alberdi</a></p>
 <p itemprop='roleName'>FW</p></div>
<div itemprop='member' itemscope itemtype='http://schema.org/OrganizationRole'>
 <p><a href='/players/spain/koldo-obieta-alberdi'>Koldo Obieta Alberdi</a></p>
 <p itemprop='roleName'>FW</p></div>
<div itemprop='member' itemscope itemtype='http://schema.org/OrganizationRole'>
 <p><a href='/players/indonesia/diego-michiels'>Diego Michiels</a></p>
 <p itemprop='roleName'>DF</p></div>
</body></html>
"""

PLAYER_HTML = """
<html><body>
<h1>Mariano Peralta Bauer Stats - Goals, xG &amp; Career Stats</h1>
<h2>This Season (2025/2026) &amp; Career Stats</h2>
<h2>Liga 1 Stats for Mariano Peralta Bauer</h2>
<table><tr><th>General</th><th>Total</th></tr>
<tr><td>Matches Played</td><td>34</td></tr>
<tr><td>Minutes</td><td>3040</td></tr></table>
<table><tr><th>Goals, xG, Shots</th><th>Total</th></tr>
<tr><td>Goals Scored</td><td>20</td></tr>
<tr><td>Expected Goals (xG)</td><td>11.52</td></tr></table>
<table><tr><td>Assists</td><td>14</td></tr></table>
<h2>President's Cup Stats for Mariano Peralta Bauer</h2>
<table><tr><td>Matches Played</td><td>2</td></tr><tr><td>Goals Scored</td><td>1</td></tr></table>
<section id='past-section'>
 <h5>2024/2025 Season</h5>
 <div class='player_season_table'>
  <div class='w100 cf player_season_row'>
   <div class='col-lg-4'><a href='/indonesia/liga-1'>Liga 1</a></div>
   <div class='col-lg-1'><p>33</p></div><div class='col-lg-1'><p>9</p></div>
   <div class='col-lg-1'><p>12</p></div><div class='col-lg-1'><p>3</p></div>
   <div class='col-lg-1'><p>0</p></div><div class='col-lg-1'><p>0</p></div>
   <div class='col-lg-2'><p>2894'</p></div>
  </div>
  <div class='w100 cf totals'><p>Total 2024/2025</p><p>33</p></div>
 </div>
 <h5>2022 Season</h5>
 <div class='player_season_table'>
  <div class='w100 cf player_season_row'>
   <div class='col-lg-4'><a href='/argentina/primera-division'>Primera División</a></div>
   <div class='col-lg-1'><p>31</p></div><div class='col-lg-1'><p>2</p></div>
   <div class='col-lg-1'><p>1</p></div><div class='col-lg-1'><p>1</p></div>
   <div class='col-lg-1'><p>0</p></div><div class='col-lg-1'><p>0</p></div>
   <div class='col-lg-2'><p>1220'</p></div>
  </div>
 </div>
</section>
</body></html>
"""


def test_season_key_web():
    assert season_key_web("2024/2025") == "2024-25"
    assert season_key_web("2024") == "2024"
    assert season_key_web(" 2019/2020 ") == "2019-20"


def test_league_season_from_title():
    assert league_season(LEAGUE_HTML) == "2025-26"


def test_league_table_rows_and_lock():
    t = parse_league_table(LEAGUE_HTML)
    assert list(t["club_raw"]) == ["Borneo FC", "Persib"]
    assert t.loc[0, "footystats_team_id"] == 1536
    assert t.loc[0, "points"] == 79 and t.loc[1, "goals_for"] == 70
    assert t.loc[0, "club_url"] == "/clubs/borneo-fc-1536"


def test_squad_dedups_double_render():
    squad = parse_squad(SQUAD_HTML)
    assert len(squad) == 2
    assert squad[0] == {
        "player_url": "/players/spain/koldo-obieta-alberdi",
        "player_raw": "Koldo Obieta Alberdi",
        "position": "FW",
    }
    assert squad[1]["position"] == "DF"


def test_career_maps_known_competitions_only():
    c = parse_career(PLAYER_HTML)
    liga1 = c[c["competition"] == "liga1"].iloc[0]
    assert liga1["season"] == "2024-25"
    assert liga1["appearances"] == 33 and liga1["minutes"] == 2894
    assert liga1["assists"] == 12
    other = c[c["competition"].isna()].iloc[0]
    # verbatim, correctly decoded, never force-mapped
    assert other["competition_raw"] == "Primera División"
    assert other["season"] == "2022" and other["minutes"] == 1220


def test_current_season_scoped_to_recognised_sections():
    cur = parse_current_season(PLAYER_HTML)
    assert len(cur) == 1  # President's Cup section is not a tracked competition
    assert cur[0] == {
        "season": "2025-26",
        "competition": "liga1",
        "appearances": 34,
        "minutes": 3040,
        "goals": 20,
        "xg": 11.52,
        "assists": 14,
    }


class CannedFetcher:
    """Serves the fixtures by URL shape; records what was requested."""

    def __init__(self):
        self.urls = []

    def fetch(self, url, key):
        self.urls.append(url)
        if "/clubs/" in url:
            return SQUAD_HTML
        if "/players/" in url:
            return PLAYER_HTML
        return LEAGUE_HTML


def test_ingest_writes_partitions(tmp_path):
    store = CorpusStore(tmp_path)
    web = FootyStatsWeb(store=store, fetcher=CannedFetcher())
    summary = web.ingest(competitions=["liga1"], max_clubs=1, max_players_per_club=1)
    assert summary.loc[0, "status"].startswith("ok")

    clubs = store.read("club_seasons", source="footystats_web")
    assert len(clubs) == 2 and set(clubs["season"]) == {"2025-26"}

    players = store.read("player_seasons", source="footystats_web")
    # one player: 2024-25 career row + 2025-26 current row (liga1 only)
    assert set(zip(players["competition"], players["season"])) == {
        ("liga1", "2024-25"),
        ("liga1", "2025-26"),
    }
    current = players[players["season"] == "2025-26"].iloc[0]
    assert current["club_raw"] == "Borneo FC" and current["position"] == "FW"
    assert current["goals"] == 20
    past = players[players["season"] == "2024-25"].iloc[0]
    assert past["club_raw"] == ""  # site does not attribute past seasons to clubs
    # provenance points at the player page, not the league page
    assert "/players/" in past["source_url"]


def test_ingest_feeds_percentile_engine(tmp_path):
    """End-to-end: scraped shape drops straight into the percentile engine."""
    from footy.corpus.percentiles import percentile

    store = CorpusStore(tmp_path)
    web = FootyStatsWeb(store=store, fetcher=CannedFetcher())
    web.ingest(competitions=["liga1"], max_clubs=1, max_players_per_club=1)
    players = store.read("player_seasons", source="footystats_web")
    # population of 1 must abstain - engine and scraper agree on column names
    assert percentile(players, "goals_per90", value=0.5, competition="liga1") is None


LEAGUE2_HTML = LEAGUE_HTML.replace("Liga 1", "Liga 2").replace(
    "borneo-fc-1536", "psms-medan-9001"
).replace("data-team-id='1536'", "data-team-id='9001'").replace("Borneo FC", "PSMS Medan")

SQUAD2_HTML = SQUAD_HTML.replace(
    "/players/spain/koldo-obieta-alberdi", "/players/indonesia/liga2-player"
).replace("Koldo Obieta Alberdi", "Liga2 Player")


class TwoLeagueFetcher:
    """liga1 and liga2 crawls whose players share career seasons."""

    def fetch(self, url, key):
        if "liga-2" in url:
            return LEAGUE2_HTML
        if "/clubs/psms" in url or "/clubs/persib" in url:
            return SQUAD2_HTML
        if "/clubs/" in url:
            return SQUAD_HTML
        if "/players/" in url:
            return PLAYER_HTML  # every player: liga1 2024-25 career row
        return LEAGUE_HTML


def test_second_league_does_not_clobber_first(tmp_path):
    """liga2 players' past Liga 1 rows must MERGE into liga1 partitions.

    Regression: writing per league let the liga2 pass replace liga1
    partitions with only its own career-spillover rows (observed live:
    a 516-row partition shrank to 32)."""
    store = CorpusStore(tmp_path)
    web = FootyStatsWeb(store=store, fetcher=TwoLeagueFetcher())
    web.ingest(competitions=["liga1", "liga2"], max_clubs=1, max_players_per_club=1)

    past = store.read("player_seasons", source="footystats_web", season="2024-25")
    # one liga1-crawled player + one liga2-crawled player, both present
    assert len(past) == 2
    assert set(past["player_url"]) == {
        "https://footystats.org/players/spain/koldo-obieta-alberdi",
        "https://footystats.org/players/indonesia/liga2-player",
    }


def test_same_player_in_two_squads_deduped(tmp_path):
    """A mover listed by two clubs contributes each identical stat line once."""

    class SameSquadFetcher(TwoLeagueFetcher):
        def fetch(self, url, key):
            if "/clubs/" in url:
                return SQUAD_HTML  # both leagues' clubs list the same player
            return super().fetch(url, key)

    store = CorpusStore(tmp_path)
    web = FootyStatsWeb(store=store, fetcher=SameSquadFetcher())
    web.ingest(competitions=["liga1", "liga2"], max_clubs=1, max_players_per_club=1)
    past = store.read("player_seasons", source="footystats_web", season="2024-25")
    assert len(past) == 1
