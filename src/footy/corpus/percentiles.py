"""The percentile engine: a raw metric -> its place in the league's population.

This is the product. "6 goals" is a number; "6 goals is the 84th percentile for
a Liga 1 midfielder with 1000+ minutes" is an insight. The dashboard is a thin
layer over this function.

Design rules:
- The comparison population is explicit and returned with every answer (n,
  filters, minutes floor). A percentile against 12 players is a different
  product from one against 400, and the caller gets to know which they got.
- Below MIN_POPULATION the engine abstains (returns None) instead of serving
  a percentile that would read as authoritative and be noise.
- Per-90 metrics are derived here, not stored: '<metric>_per90' works for any
  count column, using each player's own minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_POPULATION = 30
DEFAULT_MIN_MINUTES = 450  # ~5 full matches: below this, rates are noise


@dataclass(frozen=True)
class PercentileResult:
    percentile: float  # 0-100, share of population strictly below + half ties
    value: float
    metric: str
    n: int
    median: float
    p25: float
    p75: float
    competition: str | None
    seasons: tuple[str, ...] | None
    position: str | None
    min_minutes: int

    def sentence(self) -> str:
        pop = f"{self.n} {self.position or 'player'}s" + (
            f" in {self.competition}" if self.competition else ""
        )
        return (
            f"{self.value:g} {self.metric} is the {self.percentile:.0f}th percentile "
            f"among {pop} with {self.min_minutes}+ minutes (median {self.median:g})."
        )


def _series(pop: pd.DataFrame, metric: str) -> pd.Series:
    """Metric column, deriving '<count>_per90' from counts + minutes on demand."""
    if metric in pop.columns:
        return pd.to_numeric(pop[metric], errors="coerce")
    if metric.endswith("_per90"):
        base = metric[: -len("_per90")]
        if base in pop.columns:
            minutes = pd.to_numeric(pop["minutes"], errors="coerce")
            counts = pd.to_numeric(pop[base], errors="coerce")
            return (counts / minutes.replace(0, np.nan)) * 90.0
    raise KeyError(f"metric {metric!r} not in corpus and not derivable as per-90")


def population(
    player_seasons: pd.DataFrame,
    competition: str | None = None,
    seasons: list[str] | None = None,
    position: str | None = None,
    min_minutes: int = DEFAULT_MIN_MINUTES,
) -> pd.DataFrame:
    pop = player_seasons
    if competition is not None:
        pop = pop[pop["competition"] == competition]
    if seasons is not None:
        pop = pop[pop["season"].isin(seasons)]
    if position is not None:
        pop = pop[pop["position"].str.lower() == position.lower()]
    minutes = pd.to_numeric(pop["minutes"], errors="coerce")
    return pop[minutes >= min_minutes]


def percentile(
    player_seasons: pd.DataFrame,
    metric: str,
    value: float,
    competition: str | None = None,
    seasons: list[str] | None = None,
    position: str | None = None,
    min_minutes: int = DEFAULT_MIN_MINUTES,
) -> PercentileResult | None:
    """Empirical percentile of `value` against the filtered population.

    Returns None when the population is too small to mean anything - the
    caller shows "not enough data", never a fake percentile.
    """
    pop = population(player_seasons, competition, seasons, position, min_minutes)
    values = _series(pop, metric).dropna()
    if len(values) < MIN_POPULATION:
        return None
    below = float((values < value).sum())
    ties = float((values == value).sum())
    pct = 100.0 * (below + 0.5 * ties) / len(values)
    return PercentileResult(
        percentile=pct,
        value=float(value),
        metric=metric,
        n=int(len(values)),
        median=float(values.median()),
        p25=float(values.quantile(0.25)),
        p75=float(values.quantile(0.75)),
        competition=competition,
        seasons=tuple(seasons) if seasons else None,
        position=position,
        min_minutes=min_minutes,
    )
