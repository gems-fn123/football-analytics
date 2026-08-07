"""FootyStats API ingestor: matches and player-seasons per league season.

Licensed source (api.football-data-api.com, key-gated): the player-level layer
of the corpus. The account must have the target leagues "chosen" in the
FootyStats dashboard before data endpoints return anything - league-list still
works either way, so `available_seasons` is the preflight check.

The API key never enters provenance URLs or snapshot filenames; provenance
records the endpoint + season id, the raw snapshot records the response body.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from footy.corpus.ingest.base import PoliteFetcher
from footy.corpus.provenance import stamp
from footy.corpus.store import CorpusStore
from footy.logging_utils import get_logger

INGESTOR_VERSION = "0.1.0"
LICENCE = "footystats-api"
SOURCE = "footystats"
BASE = "https://api.football-data-api.com"

log = get_logger("footy.corpus.footystats")

COMPETITIONS = {
    "Indonesia Liga 1": "liga1",
    "Indonesia Liga 2": "liga2",
}


def api_key() -> str:
    """FOOTYSTATS_API_KEY from the environment, falling back to repo .env."""
    key = os.environ.get("FOOTYSTATS_API_KEY")
    if key:
        return key
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("FOOTYSTATS_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("FOOTYSTATS_API_KEY not set (env or .env)")


def season_key(year: int | str) -> str:
    """FootyStats year field -> our season key: 2017 -> '2017', 20212022 -> '2021-22'."""
    s = str(year)
    if len(s) == 8:
        return f"{s[:4]}-{s[6:8]}"
    return s


class FootyStats:
    def __init__(
        self, store: CorpusStore | None = None, fetcher: PoliteFetcher | None = None
    ) -> None:
        self.key = api_key()
        self.store = store or CorpusStore()
        # respect_robots=False: key-authenticated API under its own ToS; the
        # host's blanket robots Disallow targets crawlers, not paying clients.
        self.fetcher = fetcher or PoliteFetcher(SOURCE, min_interval_s=2.0, respect_robots=False)

    def _get(self, endpoint: str, snapshot_key: str, **params) -> dict:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = (
            f"{BASE}/{endpoint}?key={self.key}&{qs}" if qs else f"{BASE}/{endpoint}?key={self.key}"
        )
        text = self.fetcher.fetch(url, key=snapshot_key)
        blob = json.loads(text)
        if not blob.get("success", False):
            raise RuntimeError(f"{endpoint} {qs}: {blob.get('message', 'no message')[:160]}")
        return blob

    def _get_paged(self, endpoint: str, snapshot_prefix: str, **params) -> list[dict]:
        rows, page = [], 1
        while True:
            blob = self._get(endpoint, f"{snapshot_prefix}_p{page}", page=page, **params)
            rows.extend(blob.get("data", []))
            pager = blob.get("pager", {})
            if page >= int(pager.get("max_page", 1)):
                return rows
            page += 1

    def provenance_url(self, endpoint: str, season_id: int) -> str:
        """What goes in provenance: endpoint + season, never the key."""
        return f"{BASE}/{endpoint}?season_id={season_id}"

    def available_seasons(self) -> pd.DataFrame:
        """Indonesian seasons visible to this key, with chosen-status preflight."""
        every = self._get("league-list", "league_list")["data"]
        chosen = self._get("league-list", "league_list_chosen", chosen_leagues_only="true")["data"]
        chosen_names = {x.get("name") for x in chosen}
        rows = []
        for league in every:
            comp = COMPETITIONS.get(league.get("name", ""))
            if not comp:
                continue
            for season in league.get("season", []):
                rows.append(
                    {
                        "competition": comp,
                        "league_name": league["name"],
                        "season": season_key(season.get("year", "")),
                        "season_id": int(season["id"]),
                        "chosen": league["name"] in chosen_names,
                    }
                )
        return pd.DataFrame(rows).sort_values(["competition", "season"]).reset_index(drop=True)

    # -- normalisers (pure; unit-tested against recorded fixtures) ------------

    @staticmethod
    def normalise_matches(raw: list[dict]) -> pd.DataFrame:
        rows = []
        for m in raw:
            if m.get("status") != "complete":
                continue
            rows.append(
                {
                    "footystats_match_id": int(m["id"]),
                    "date": (
                        pd.Timestamp(int(m["date_unix"]), unit="s", tz="UTC").date().isoformat()
                        if m.get("date_unix")
                        else None
                    ),
                    "home_raw": str(m.get("home_name", "")),
                    "away_raw": str(m.get("away_name", "")),
                    "home_goals": int(m.get("homeGoalCount", 0)),
                    "away_goals": int(m.get("awayGoalCount", 0)),
                    "home_footystats_id": m.get("homeID"),
                    "away_footystats_id": m.get("awayID"),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def normalise_players(raw: list[dict], club_names: dict[int, str]) -> pd.DataFrame:
        rows = []
        for p in raw:
            club_id = p.get("club_team_id")
            rows.append(
                {
                    "footystats_player_id": int(p["id"]),
                    "player_raw": str(p.get("full_name", "")),
                    "known_as_raw": str(p.get("known_as", "")),
                    "position": str(p.get("position") or ""),
                    "club_raw": club_names.get(club_id, ""),
                    "club_footystats_id": club_id,
                    "appearances": p.get("appearances_overall"),
                    "minutes": p.get("minutes_played_overall"),
                    "goals": p.get("goals_overall"),
                    "assists": p.get("assists_overall"),
                    "yellow_cards": p.get("yellow_cards_overall"),
                    "red_cards": p.get("red_cards_overall"),
                    "clean_sheets": p.get("clean_sheets_overall"),
                }
            )
        return pd.DataFrame(rows)

    # -- ingestion ------------------------------------------------------------

    def ingest_season(self, competition: str, season: str, season_id: int) -> dict:
        matches_raw = self._get_paged(
            "league-matches", f"{competition}_{season}_matches", season_id=season_id
        )
        teams_raw = self._get_paged(
            "league-teams", f"{competition}_{season}_teams", season_id=season_id
        )
        players_raw = self._get_paged(
            "league-players", f"{competition}_{season}_players", season_id=season_id
        )
        club_names = {t.get("id"): str(t.get("name", "")) for t in teams_raw}
        meta = {"competition": competition, "season": season}

        matches = self.normalise_matches(matches_raw).assign(**meta)
        players = self.normalise_players(players_raw, club_names).assign(**meta)
        counts = {"matches": len(matches), "players": len(players)}

        for table, df, endpoint in (
            ("matches", matches, "league-matches"),
            ("player_seasons", players, "league-players"),
        ):
            if not len(df):
                continue
            df = stamp(
                df,
                source=SOURCE,
                source_url=self.provenance_url(endpoint, season_id),
                licence_tag=LICENCE,
                ingestor_version=INGESTOR_VERSION,
            )
            self.store.write(table, df, source=SOURCE, competition=competition, season=season)
        return counts

    def ingest_all(self) -> pd.DataFrame:
        """Backfill every chosen Indonesian season. Skips (and reports) unchosen ones."""
        seasons = self.available_seasons()
        summary = []
        for _, row in seasons.iterrows():
            if not row["chosen"]:
                summary.append({**row.to_dict(), "status": "NOT CHOSEN in dashboard"})
                continue
            try:
                counts = self.ingest_season(row["competition"], row["season"], row["season_id"])
                summary.append({**row.to_dict(), "status": "ok", **counts})
            except Exception as exc:
                log.warning("season %s failed: %s", row["season_id"], exc)
                summary.append({**row.to_dict(), "status": str(exc)[:100]})
        return pd.DataFrame(summary)
