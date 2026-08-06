"""Partitioned Parquet store for the corpus.

Layout: <root>/<table>/source=<source>/competition=<comp>/season=<season>/data.parquet

Partitioning by source first is deliberate: if a source's terms or legal status
change, `drop_source` removes every trace in one call and downstream aggregates
are rebuilt from what remains. Writes are idempotent per partition (re-ingesting
a season replaces it), so ingestors can be re-run safely at any time.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

from footy.corpus.provenance import PROVENANCE
from footy.logging_utils import get_logger

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(value: str) -> str:
    """Filesystem-safe partition value; keeps the raw value readable."""
    return _SAFE.sub("_", value.strip()) or "unknown"


class CorpusStore:
    def __init__(self, root: str | Path = "data/processed/corpus") -> None:
        self.root = Path(root)
        self.log = get_logger("footy.corpus.store")

    def _partition_dir(self, table: str, source: str, competition: str, season: str) -> Path:
        return (
            self.root
            / _safe(table)
            / f"source={_safe(source)}"
            / f"competition={_safe(competition)}"
            / f"season={_safe(season)}"
        )

    def write(
        self, table: str, df: pd.DataFrame, *, source: str, competition: str, season: str
    ) -> Path:
        """Replace one partition. Every row must carry full provenance."""
        missing = set(PROVENANCE) - set(df.columns)
        if missing:
            raise ValueError(f"{table}: rows missing provenance columns {sorted(missing)}")
        part = self._partition_dir(table, source, competition, season)
        part.mkdir(parents=True, exist_ok=True)
        out = part / "data.parquet"
        df.to_parquet(out, index=False)
        self.log.info("wrote %s rows=%d", out, len(df))
        return out

    def read(
        self,
        table: str,
        *,
        source: str | None = None,
        competition: str | None = None,
        season: str | None = None,
    ) -> pd.DataFrame:
        """Concatenate matching partitions; empty frame when nothing matches."""
        base = self.root / _safe(table)
        if not base.exists():
            return pd.DataFrame()
        frames = []
        for f in sorted(base.rglob("data.parquet")):
            parts = dict(p.split("=", 1) for p in f.parent.parts if "=" in p)
            if source and parts.get("source") != _safe(source):
                continue
            if competition and parts.get("competition") != _safe(competition):
                continue
            if season and parts.get("season") != _safe(season):
                continue
            frames.append(pd.read_parquet(f))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def drop_source(self, source: str) -> int:
        """Quarantine: remove every partition of a source, across all tables."""
        n = 0
        for d in self.root.glob(f"*/source={_safe(source)}"):
            shutil.rmtree(d)
            n += 1
        self.log.warning("dropped source=%s (%d table partitions)", source, n)
        return n

    def manifest(self) -> pd.DataFrame:
        """One row per partition: table, source, competition, season, rows."""
        rows = []
        for f in sorted(self.root.rglob("data.parquet")):
            parts = dict(p.split("=", 1) for p in f.parent.parts if "=" in p)
            rows.append(
                {
                    "table": f.parent.parent.parent.parent.name,
                    "source": parts.get("source"),
                    "competition": parts.get("competition"),
                    "season": parts.get("season"),
                    "rows": len(pd.read_parquet(f, columns=["source"])),
                }
            )
        return pd.DataFrame(rows)
