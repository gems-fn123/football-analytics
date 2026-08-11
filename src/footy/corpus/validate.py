"""Cross-source validation: do independent sources tell the same story?

Two scrapers agreeing on 34 matches' worth of W/D/L/GF/GA is strong evidence
both parsed correctly; a disagreement is either a parser bug (found one: the
Wikipedia ingestor NaN'd every footnoted Pts cell) or a genuine editorial
difference worth knowing about (found one of those too: FootyStats orders
points-ties by goal difference, Wikipedia records the official head-to-head
standings - so `position` is compared but reported separately from the
factual columns).
"""

from __future__ import annotations

import pandas as pd

from footy.corpus.entities import ClubResolver
from footy.corpus.store import CorpusStore

FACT_COLUMNS = ["played", "won", "drawn", "lost", "goals_for", "goals_against", "points"]


def cross_validate_club_seasons(
    store: CorpusStore,
    source_a: str,
    source_b: str,
    competition: str,
    season: str,
    resolver: ClubResolver | None = None,
) -> pd.DataFrame:
    """One row per (club, field) where the two sources disagree.

    Facts (played/won/.../points) and `position` are both checked; position
    disagreements are tagged kind='ordering' because league tie-break order
    is editorial, while the counting columns are tagged kind='fact'.
    Clubs present in only one source appear as kind='coverage'.
    """
    resolver = resolver or ClubResolver()
    frames = {}
    for src in (source_a, source_b):
        df = store.read("club_seasons", source=src, competition=competition, season=season)
        if not len(df):
            return pd.DataFrame(
                [{"club_id": None, "field": "table", "kind": "coverage",
                  source_a: "present" if src != source_a else "MISSING",
                  source_b: "present" if src != source_b else "MISSING"}]
            )
        name_col = "club_name" if "club_name" in df.columns else "club_raw"
        df = df.copy()
        df["club_id"] = df[name_col].map(resolver.resolve)
        frames[src] = df

    # Unresolved clubs cannot be paired (and None merges with None into a
    # cartesian mess); they are reported once each, as their raw string.
    diffs: list[dict] = []
    for src in (source_a, source_b):
        df = frames[src]
        name_col = "club_name" if "club_name" in df.columns else "club_raw"
        for raw in sorted(df.loc[df["club_id"].isna(), name_col].unique()):
            diffs.append(
                {"club_id": f"?{raw}", "field": "club", "kind": "unresolved",
                 source_a: "this source" if src == source_a else "",
                 source_b: "this source" if src == source_b else ""}
            )
    a = frames[source_a].dropna(subset=["club_id"])
    b = frames[source_b].dropna(subset=["club_id"])

    ids_a, ids_b = set(a["club_id"]), set(b["club_id"])
    for cid in sorted(ids_a ^ ids_b):
        diffs.append(
            {"club_id": cid, "field": "club", "kind": "coverage",
             source_a: "present" if cid in ids_a else "missing",
             source_b: "present" if cid in ids_b else "missing"}
        )

    merged = a.merge(b, on="club_id", suffixes=("_a", "_b"))
    for col, kind in [(c, "fact") for c in FACT_COLUMNS] + [("position", "ordering")]:
        ca, cb = f"{col}_a", f"{col}_b"
        if ca not in merged.columns or cb not in merged.columns:
            continue
        va = pd.to_numeric(merged[ca], errors="coerce")
        vb = pd.to_numeric(merged[cb], errors="coerce")
        for _, row in merged[(va != vb) & ~(va.isna() & vb.isna())].iterrows():
            diffs.append(
                {"club_id": row["club_id"], "field": col, "kind": kind,
                 source_a: row[ca], source_b: row[cb]}
            )
    return pd.DataFrame(diffs, columns=["club_id", "field", "kind", source_a, source_b])
