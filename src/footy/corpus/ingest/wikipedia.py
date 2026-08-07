"""Wikipedia season-page ingestor: league tables and result grids, 2017 onward.

Wikipedia is the cleanest free backbone: CC-BY-SA, stable, and it documents the
club renames and competition rebrands that make Indonesian football hard
(Liga 1 -> Super League in 2025, operator LIB -> I-League). Facts extracted:
final league tables (club-season aggregates) and the home x away results grid.

Names are stored VERBATIM (club_raw / away_raw); resolving "Persib" ==
"Persib Bandung" is entity resolution's job (step 3), never the scraper's.
"""

from __future__ import annotations

import re
from io import StringIO
from urllib.parse import quote

import numpy as np
import pandas as pd

from footy.corpus.ingest.base import PoliteFetcher
from footy.corpus.provenance import stamp
from footy.corpus.store import CorpusStore
from footy.logging_utils import get_logger

INGESTOR_VERSION = "0.1.0"
LICENCE = "CC-BY-SA-4.0"
SOURCE = "wikipedia"

log = get_logger("footy.corpus.wikipedia")

# Internal competition keys are stable across rebrands; the page title records
# what the competition was called that season.
SEASON_PAGES: dict[str, list[tuple[str, str]]] = {
    "liga1": [
        ("2017", "2017 Liga 1 (Indonesia)"),
        ("2018", "2018 Liga 1 (Indonesia)"),
        ("2019", "2019 Liga 1 (Indonesia)"),
        ("2020", "2020 Liga 1 (Indonesia)"),  # abandoned (COVID); table still published
        ("2021-22", "2021–22 Liga 1 (Indonesia)"),
        ("2022-23", "2022–23 Liga 1 (Indonesia)"),
        ("2023-24", "2023–24 Liga 1 (Indonesia)"),
        ("2024-25", "2024–25 Liga 1 (Indonesia)"),
        ("2025-26", "2025–26 Super League (Indonesia)"),
        ("2026-27", "2026–27 Super League (Indonesia)"),
    ],
    "liga2": [
        ("2017", "2017 Liga 2 (Indonesia)"),
        ("2018", "2018 Liga 2 (Indonesia)"),
        ("2019", "2019 Liga 2 (Indonesia)"),
        ("2020", "2020 Liga 2 (Indonesia)"),
        ("2021-22", "2021–22 Liga 2 (Indonesia)"),
        ("2022-23", "2022–23 Liga 2 (Indonesia)"),
        ("2023-24", "2023–24 Liga 2 (Indonesia)"),
        ("2024-25", "2024–25 Liga 2 (Indonesia)"),
        ("2025-26", "2025–26 Championship (Indonesia)"),
    ],
}

_SCORE = re.compile(r"^\s*(\d+)\s*[–—-]\s*(\d+)")
_MARKERS = re.compile(r"\s*(\((?:C|R|Q|O|A|P|X)\)|\[[a-z0-9]+\])\s*$", re.IGNORECASE)


def page_url(title: str) -> str:
    # /wiki/ article URLs, not the REST API: en.wikipedia.org's robots.txt
    # disallows /api/ for generic agents and PoliteFetcher takes robots at its
    # word - no special-casing "but the API is meant for bots".
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="_()")


def strip_markers(name: str) -> str:
    """Remove champion/relegation/footnote suffixes: 'Bali United (C)[a]' -> 'Bali United'."""
    out = str(name).strip()
    while True:
        new = _MARKERS.sub("", out)
        if new == out:
            return out
        out = new


def _sections(html: str) -> list[tuple[str, pd.DataFrame]]:
    """(section label, table) for every wikitable, in document order.

    Liga 2 runs in stages and regional groups (First round West/East, Second
    round Group A/B); the group is in the h2/h3 headings ABOVE each table,
    which pd.read_html discards. So: walk the DOM tracking the heading path,
    then hand each table to pandas individually. html5lib parser throughout -
    live Wikipedia HTML carries malformed attributes (rowspan='2style=...')
    that abort lxml mid-page.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html5lib")
    path = {"h2": "", "h3": ""}
    out: list[tuple[str, pd.DataFrame]] = []
    for el in soup.find_all(["h2", "h3", "table"]):
        if el.name in ("h2", "h3"):
            txt = re.sub(r"\[\s*edit\s*\]\s*$", "", el.get_text(" ", strip=True)).strip()
            path[el.name] = txt
            if el.name == "h2":
                path["h3"] = ""
        else:
            classes = el.get("class") or []
            if "wikitable" not in classes:
                continue
            label = " / ".join(x for x in (path["h2"], path["h3"]) if x)
            try:
                dfs = pd.read_html(StringIO(str(el)), flavor="bs4")
            except ValueError:
                continue
            out.extend((label, df) for df in dfs)
    return out


def parse_league_table(html: str) -> pd.DataFrame | None:
    """EVERY standings-shaped table on the page -> club rows with a group_raw
    column carrying the section label ('First round / West region', ...).
    A club appearing in several stages yields one row per stage - different
    facts, not duplicates."""
    frames = []
    for label, t in _sections(html):
        cols = [str(c) for c in t.columns]
        if not any(c.startswith("Pos") for c in cols):
            continue
        team_col = next((c for c in t.columns if str(c).startswith("Team")), None)
        if team_col is None or not {"Pld", "Pts"} <= set(cols):
            continue
        out = pd.DataFrame(
            {
                "position": pd.to_numeric(t["Pos"], errors="coerce"),
                "club_raw": t[team_col].astype(str),
                "club_name": t[team_col].astype(str).map(strip_markers),
                "played": pd.to_numeric(t["Pld"], errors="coerce"),
                "won": pd.to_numeric(t.get("W"), errors="coerce"),
                "drawn": pd.to_numeric(t.get("D"), errors="coerce"),
                "lost": pd.to_numeric(t.get("L"), errors="coerce"),
                "goals_for": pd.to_numeric(t.get("GF"), errors="coerce"),
                "goals_against": pd.to_numeric(t.get("GA"), errors="coerce"),
                "points": pd.to_numeric(t.get("Pts"), errors="coerce"),
                "group_raw": label,
            }
        ).dropna(subset=["position", "played"])
        if len(out):
            frames.append(out)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def parse_results_grid(html: str) -> pd.DataFrame | None:
    """Every Home x Away score matrix on the page -> one row per played
    fixture, tagged with the section label in group_raw.

    Column headers are club short forms; row labels are fuller names. Both are
    kept verbatim for entity resolution. Unplayed/void cells are skipped.
    """
    frames = []
    for label, t in _sections(html):
        first = str(t.columns[0])
        if "Home" not in first or "Away" not in first:
            continue
        rows = []
        for _, r in t.iterrows():
            home_raw = str(r.iloc[0])
            for abbr in t.columns[1:]:
                m = _SCORE.match(str(r[abbr]))
                if m:
                    rows.append(
                        {
                            "home_raw": home_raw,
                            "home_name": strip_markers(home_raw),
                            "away_raw": str(abbr),
                            "home_goals": int(m.group(1)),
                            "away_goals": int(m.group(2)),
                            "group_raw": label,
                        }
                    )
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def ingest(
    store: CorpusStore | None = None,
    competitions: list[str] | None = None,
    fetcher: PoliteFetcher | None = None,
) -> pd.DataFrame:
    """Backfill league tables + result grids. Returns a summary frame."""
    store = store or CorpusStore()
    fetcher = fetcher or PoliteFetcher(SOURCE, min_interval_s=1.0)
    summary = []
    for comp in competitions or list(SEASON_PAGES):
        for season, title in SEASON_PAGES[comp]:
            url = page_url(title)
            try:
                html = fetcher.fetch(url, key=f"{comp}_{season}")
            except Exception as exc:
                log.warning("skip %s %s: %s", comp, season, exc)
                summary.append({"competition": comp, "season": season, "status": str(exc)[:80]})
                continue

            meta = {
                "competition": comp,
                "season": season,
                "competition_name_raw": title,
            }
            n_clubs = n_matches = 0

            table = parse_league_table(html)
            if table is not None:
                table = table.assign(**meta)
                table = stamp(
                    table,
                    source=SOURCE,
                    source_url=url,
                    licence_tag=LICENCE,
                    ingestor_version=INGESTOR_VERSION,
                )
                store.write("club_seasons", table, source=SOURCE, competition=comp, season=season)
                n_clubs = len(table)

            grid = parse_results_grid(html)
            if grid is not None:
                grid = grid.assign(**meta)
                grid["home_goals"] = grid["home_goals"].astype(np.int16)
                grid["away_goals"] = grid["away_goals"].astype(np.int16)
                grid = stamp(
                    grid,
                    source=SOURCE,
                    source_url=url,
                    licence_tag=LICENCE,
                    ingestor_version=INGESTOR_VERSION,
                )
                store.write("matches", grid, source=SOURCE, competition=comp, season=season)
                n_matches = len(grid)

            summary.append(
                {
                    "competition": comp,
                    "season": season,
                    "status": "ok",
                    "clubs": n_clubs,
                    "matches": n_matches,
                }
            )
    return pd.DataFrame(summary)
