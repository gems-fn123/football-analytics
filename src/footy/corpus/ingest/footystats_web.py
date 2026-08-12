"""footystats.org website scraper: league tables, squads, player career stats.

This is the free-tier sibling of the FootyStats API ingestor. The website
serves Liga 1/Liga 2 pages fully server-rendered to our honest user agent,
and robots.txt allows the paths we crawl (league pages, /clubs/*, /players/*).
The paths robots.txt disallows - /players?* search, /c-dl.php CSV downloads,
/api/* - are respected and never fetched, even though the CSV endpoint would
be the easier route.

Crawl shape (one league season, ~1 + clubs + squad-size requests):
    league page   -> league table (club aggregates) + club page URLs
    club pages    -> squad list: player page URLs + positions
    player pages  -> "Previous Seasons' Performances" (per season x competition:
                     MP, goals, assists, cards, penalties, minutes) plus the
                     current-season stat tables

Facts only this source has for free: player minutes and appearances, which is
what the percentile engine's minutes floor needs.

Honesty notes, per the "nullable over guessed" rule:
- Career rows carry no club column on the site, so club_raw is empty for past
  seasons; only the current-season row is attributed to the club whose squad
  page listed the player (current roster, so mid-season movers may be listed
  under their newest club).
- Position comes from the current squad listing and is applied to all of a
  player's rows; it is a player attribute, not a season fact, on this site.
"""

from __future__ import annotations

import re

import pandas as pd
from bs4 import BeautifulSoup

from footy.corpus.ingest.base import PoliteFetcher
from footy.corpus.provenance import stamp
from footy.corpus.store import CorpusStore
from footy.logging_utils import get_logger

INGESTOR_VERSION = "0.2.0"  # 0.2.0: + xA, age/birth date from the profile box
LICENCE = "tos-risk-accepted"  # scraped public pages; user-approved Tier-B risk
SOURCE = "footystats_web"
BASE = "https://footystats.org"

log = get_logger("footy.corpus.footystats_web")

LEAGUE_PAGES = {
    "liga1": f"{BASE}/indonesia/liga-1",
    "liga2": f"{BASE}/indonesia/liga-2",
}

# Career-table competition links -> our stable keys. Anything not listed keeps
# its verbatim name with competition=None (other leagues, cups, friendlies).
COMPETITION_PATHS = {
    "/indonesia/liga-1": "liga1",
    "/indonesia/liga-2": "liga2",
}

# Current-season stat sections: h2 "<competition> Stats for <player>".
_SECTION_COMPETITIONS = {
    "Liga 1 Stats": "liga1",
    "Liga 2 Stats": "liga2",
}

_THIS_SEASON = re.compile(r"This Season \((\d{4})/(\d{4})\)")
_TITLE_SEASON = re.compile(r"(\d{4})/(\d{2})\b")


def season_key_web(raw: str) -> str:
    """'2024/2025' -> '2024-25'; '2024' -> '2024' (calendar-year seasons)."""
    m = re.match(r"^\s*(\d{4})\s*/\s*(\d{4})\s*$", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)[2:]}"
    m = re.match(r"^\s*(\d{4})\s*$", raw)
    if m:
        return m.group(1)
    return raw.strip()


def _int(text: str | None) -> int | None:
    """Digits or None - never zero-fill. Strips the minutes apostrophe."""
    if text is None:
        return None
    digits = re.sub(r"[^\d]", "", str(text))
    return int(digits) if digits else None


def _float(text: str | None) -> float | None:
    if text is None:
        return None
    m = re.search(r"\d+(?:\.\d+)?", str(text))
    return float(m.group(0)) if m else None


def _soup(html: str) -> BeautifulSoup:
    # lxml, not html5lib: footystats markup is clean (verified identical
    # parser output on live pages), and a full crawl re-parses ~1000 snapshot
    # pages - the 3x parser speedup is minutes of wall clock.
    return BeautifulSoup(html, "lxml")


def league_season(html: str) -> str | None:
    """Season key from the league page title: '... Liga 1 2025/26 Stats'."""
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    if not m:
        return None
    t = _TITLE_SEASON.search(m.group(1))
    return f"{t.group(1)}-{t.group(2)}" if t else None


def parse_league_table(html: str) -> pd.DataFrame | None:
    """The full-league-table -> one row per club, with club page URLs."""
    table = _soup(html).find("table", class_="full-league-table")
    if table is None or table.find("tbody") is None:
        return None
    rows = []
    for tr in table.tbody.find_all("tr"):
        team_a = None
        team_td = tr.find("td", class_="team")
        if team_td is not None:
            team_a = team_td.find("a")
        if team_a is None:
            continue

        def cell(cls: str) -> int | None:
            td = tr.find("td", class_=cls)
            if td is None or td.find("a", href="/premium"):  # locked for free tier
                return None
            return _int(td.get_text(strip=True))

        pos_td = tr.find("td", class_="position")
        rows.append(
            {
                "position": _int(pos_td.get_text(strip=True)) if pos_td else None,
                "club_raw": team_a.get_text(strip=True),
                "club_name": team_a.get_text(strip=True),
                "club_url": team_a.get("href"),
                "footystats_team_id": _int(team_a.get("data-team-id")),
                "footystats_comp_id": _int(team_a.get("data-comp-id")),
                "played": cell("mp"),
                "won": cell("win"),
                "drawn": cell("draw"),
                "lost": cell("loss"),
                "goals_for": cell("gf"),
                "goals_against": cell("ga"),
                "points": cell("points"),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).dropna(subset=["position", "played"])


_AGE = re.compile(r"Age\s*:\s*(\d{1,2})\s*(?:\(([^)]+)\))?")


def parse_profile(html: str) -> dict:
    """Age and birth date from the player info box: 'Age : 28 (February 20, 1998)'.

    Both nullable; the birth date is the durable fact (age drifts with the
    retrieval date), kept ISO when it parses, verbatim when it doesn't.
    """
    flat = re.sub(r"<[^>]+>", " ", html)
    m = _AGE.search(flat)
    if not m:
        return {"age": None, "birth_date": None}
    birth = None
    if m.group(2):
        try:
            birth = pd.Timestamp(m.group(2).strip()).date().isoformat()
        except (ValueError, TypeError):
            birth = m.group(2).strip()
    return {"age": int(m.group(1)), "birth_date": birth}


def parse_squad(html: str) -> list[dict]:
    """Squad entries from a club page: player URL, verbatim name, position.

    The page renders each row twice (desktop + mobile); dedup by player URL.
    """
    seen: dict[str, dict] = {}
    for member in _soup(html).find_all(
        "div", itemtype="http://schema.org/OrganizationRole"
    ):
        a = member.find("a", href=re.compile(r"^/players/"))
        if a is None:
            continue
        url = a.get("href", "")
        if url in seen:
            continue
        role = member.find(itemprop="roleName")
        seen[url] = {
            "player_url": url,
            "player_raw": a.get_text(strip=True),
            "position": role.get_text(strip=True) if role else None,
        }
    return list(seen.values())


def parse_career(html: str) -> pd.DataFrame | None:
    """'Previous Seasons' Performances' -> one row per season x competition.

    Competition names are verbatim; the competition key is set only when the
    row's link matches COMPETITION_PATHS - other competitions stay in the
    frame with competition=None so nothing is guessed.
    """
    section = _soup(html).find("section", id="past-section")
    if section is None:
        return None
    rows = []
    for row in section.find_all("div", class_="player_season_row"):
        h5 = row.find_previous("h5")
        if h5 is None:
            continue
        season_raw = re.sub(r"\s*Season\s*$", "", h5.get_text(strip=True))
        comp_a = row.find("a")
        comp_href = comp_a.get("href", "") if comp_a else ""
        counters = [p.get_text(strip=True) for p in row.select("div.col-lg-1 p")]
        counters += [None] * (6 - len(counters))
        minutes_p = row.select_one("div.col-lg-2 p")
        rows.append(
            {
                "season_raw": season_raw,
                "season": season_key_web(season_raw),
                "competition_raw": comp_a.get_text(strip=True) if comp_a else "",
                "competition": COMPETITION_PATHS.get(comp_href),
                "appearances": _int(counters[0]),
                "goals": _int(counters[1]),
                "assists": _int(counters[2]),
                "yellow_cards": _int(counters[3]),
                "red_cards": _int(counters[4]),
                "penalties_scored": _int(counters[5]),
                "minutes": _int(minutes_p.get_text(strip=True) if minutes_p else None),
            }
        )
    if not rows:
        return None
    # Desktop/mobile double-render yields exact duplicate rows; identical
    # facts twice is an artifact, not two spells.
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def parse_current_season(html: str) -> list[dict]:
    """Current-season totals from the per-competition stat tables.

    Walks h2 headings; under 'Liga 1 Stats for <player>' collects the first
    'Matches Played' / 'Minutes' / 'Goals' / 'Assists' totals seen (the page
    repeats tables for mobile - first occurrence wins). Returns one dict per
    recognised competition, or [] when the player has no current-season block.
    """
    soup = _soup(html)
    season = None
    for h2 in soup.find_all("h2"):
        m = _THIS_SEASON.search(h2.get_text(" ", strip=True))
        if m:
            season = f"{m.group(1)}-{m.group(2)[2:]}"
            break
    if season is None:
        return []

    wanted = {
        "matches played": "appearances",
        "minutes": "minutes",
        "goals scored": "goals",
        "assists": "assists",
        "expected goals (xg)": "xg",
        "expected assists (xa)": "xa",
    }
    out: list[dict] = []
    current: dict | None = None
    for el in soup.find_all(["h2", "table"]):
        if el.name == "h2":
            label = el.get_text(" ", strip=True)
            comp = next(
                (key for prefix, key in _SECTION_COMPETITIONS.items() if label.startswith(prefix)),
                None,
            )
            if comp is not None:
                current = {"season": season, "competition": comp}
                out.append(current)
            elif re.search(r"Stats for|Previous Seasons|Squad", label):
                current = None  # left the recognised competition's section
            continue
        if current is None:
            continue
        for tr in el.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            field = wanted.get(cells[0].get_text(" ", strip=True).lower())
            if field and field not in current:
                text = cells[1].get_text(" ", strip=True)
                current[field] = _float(text) if field in ("xg", "xa") else _int(text)
    return [c for c in out if c.get("appearances") is not None]


class FootyStatsWeb:
    """Crawl footystats.org league -> clubs -> players into the corpus."""

    def __init__(
        self, store: CorpusStore | None = None, fetcher: PoliteFetcher | None = None
    ) -> None:
        self.store = store or CorpusStore()
        # 3 s measured: at 2 s footystats starts burst-429ing after ~400 pages
        self.fetcher = fetcher or PoliteFetcher(SOURCE, min_interval_s=3.0)

    def _player_frame(self, url: str, html: str, position: str | None, club_raw: str) -> pd.DataFrame:
        """All of one player's season rows (career + current), stamped with
        this player page's URL so provenance is per-record-exact."""
        name = ""
        h1 = _soup(html).find("h1")
        if h1 is not None:
            name = re.sub(r"\s+Stats.*$", "", h1.get_text(" ", strip=True))
        frames = []
        career = parse_career(html)
        if career is not None:
            career = career[career["competition"].notna()].copy()
            if len(career):
                career["club_raw"] = ""  # site does not name the club per past season
                frames.append(career)
        current = parse_current_season(html)
        if current:
            cur = pd.DataFrame(current)
            cur["season_raw"] = cur["season"]
            cur["competition_raw"] = ""
            cur["club_raw"] = club_raw
            frames.append(cur)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["player_raw"] = name
        df["player_url"] = url
        df["position"] = position
        profile = parse_profile(html)
        df["age"] = profile["age"]  # as of retrieval, not of the season row
        df["birth_date"] = profile["birth_date"]
        return stamp(
            df,
            source=SOURCE,
            source_url=url,
            licence_tag=LICENCE,
            ingestor_version=INGESTOR_VERSION,
        )

    def ingest(
        self,
        competitions: list[str] | None = None,
        max_clubs: int | None = None,
        max_players_per_club: int | None = None,
    ) -> pd.DataFrame:
        """Crawl league(s); returns a summary frame. Caps are for prototyping -
        a capped run is a sample, and the summary says so.

        Player rows are written once, after every league is crawled: a liga2
        player's past Liga 1 seasons belong in the same partition as the
        liga1 crawl's rows, and store.write REPLACES partitions - writing
        per league would let the later league clobber the earlier one's.
        """
        summary = []
        all_player_frames: list[pd.DataFrame] = []
        for comp in competitions or list(LEAGUE_PAGES):
            league_url = LEAGUE_PAGES[comp]
            html = self.fetcher.fetch(league_url, key=f"{comp}_league")
            season = league_season(html)
            table = parse_league_table(html)
            if table is None or season is None:
                summary.append({"competition": comp, "status": "league page unparseable"})
                continue

            stamped = stamp(
                table.assign(competition=comp, season=season),
                source=SOURCE,
                source_url=league_url,
                licence_tag=LICENCE,
                ingestor_version=INGESTOR_VERSION,
            )
            self.store.write("club_seasons", stamped, source=SOURCE, competition=comp, season=season)

            player_frames: list[pd.DataFrame] = []
            clubs = table.head(max_clubs) if max_clubs else table
            n_players = 0
            for _, club in clubs.iterrows():
                club_url = BASE + club["club_url"]
                try:
                    club_html = self.fetcher.fetch(
                        club_url, key=f"club_{club['footystats_team_id']}"
                    )
                except Exception as exc:
                    log.warning("club %s failed: %s", club["club_raw"], exc)
                    continue
                squad = parse_squad(club_html)
                if max_players_per_club:
                    squad = squad[:max_players_per_club]
                for entry in squad:
                    player_url = BASE + entry["player_url"]
                    try:
                        player_html = self.fetcher.fetch(
                            player_url, key="player" + entry["player_url"].replace("/", "_")
                        )
                    except Exception as exc:
                        log.warning("player %s failed: %s", entry["player_url"], exc)
                        continue
                    frame = self._player_frame(
                        player_url, player_html, entry["position"], club["club_raw"]
                    )
                    if len(frame):
                        player_frames.append(frame)
                    n_players += 1

            all_player_frames.extend(player_frames)
            summary.append(
                {
                    "competition": comp,
                    "season": season,
                    "status": "ok" + (" (capped sample)" if max_clubs or max_players_per_club else ""),
                    "clubs": len(clubs),
                    "players_fetched": n_players,
                    "player_season_rows": int(sum(len(f) for f in player_frames)),
                }
            )

        if all_player_frames:
            players = pd.concat(all_player_frames, ignore_index=True)
            # A player listed in two squads (mid-window mover) arrives twice;
            # identical stat lines are one fact. Genuine two-spell seasons
            # (loans) differ in appearances/minutes and both survive.
            players = players.drop_duplicates(
                subset=["player_url", "competition", "season", "appearances", "minutes", "goals"],
                keep="first",
            )
            for (row_comp, row_season), group in players.groupby(["competition", "season"]):
                self.store.write(
                    "player_seasons",
                    group.reset_index(drop=True),
                    source=SOURCE,
                    competition=row_comp,
                    season=row_season,
                )
        return pd.DataFrame(summary)
