"""Provenance columns stamped onto every corpus record.

Legal defensibility and debuggability both hinge on being able to say, for any
number in the corpus: where it came from, when we took it, and under what terms.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

PROVENANCE = {
    "source": "string",  # short source key, e.g. "wikipedia", "fctables"
    "source_url": "string",  # the exact page/endpoint the record came from
    "retrieved_at": "string",  # ISO 8601 UTC
    "licence_tag": "string",  # e.g. "CC-BY-SA-4.0", "tos-unreviewed", "api-licensed"
    "ingestor_version": "string",  # bump when parse logic changes materially
}


def stamp(
    df: pd.DataFrame,
    *,
    source: str,
    source_url: str,
    licence_tag: str,
    ingestor_version: str,
    retrieved_at: str | None = None,
) -> pd.DataFrame:
    """Return df with the provenance columns added (overwriting if present)."""
    out = df.copy()
    out["source"] = source
    out["source_url"] = source_url
    out["retrieved_at"] = retrieved_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["licence_tag"] = licence_tag
    out["ingestor_version"] = ingestor_version
    return out.astype(PROVENANCE)
