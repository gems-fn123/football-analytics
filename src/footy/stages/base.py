"""Stage protocol. Every stage obeys the same contract so they can be reordered,
skipped, or swapped for a different backend without touching the orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from footy.logging_utils import get_logger


@dataclass
class StageResult:
    name: str
    table: pd.DataFrame
    artifacts: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def upstream_table(ctx: dict[str, Any], *keys: str, stage: str) -> pd.DataFrame:
    """The table of the first upstream stage that actually ran.

    Stages may be skipped via the `stages:` toggles, so downstream stages declare an
    ordered preference of inputs instead of hard-keying one. If none of them ran,
    fail with an actionable message rather than a KeyError three minutes into a run.
    """
    for key in keys:
        if key in ctx:
            return ctx[key].table
    raise RuntimeError(
        f"stage {stage!r} needs one of {list(keys)} to have run; "
        "enable it under stages: in configs/pipeline.yaml"
    )


class Stage(ABC):
    name: str = "stage"
    requires_gpu: bool = False

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.log = get_logger(f"footy.{self.name}")

    # setup/teardown are deliberate optional hooks, not abstract methods: a stage that
    # holds no weights needs neither, and forcing empty overrides on every subclass is noise.
    def setup(self) -> None:  # noqa: B027
        """Load weights. Called once before the first run()."""

    @abstractmethod
    def run(self, ctx: dict[str, Any]) -> StageResult:
        """ctx carries the outputs of earlier stages, keyed by stage name."""

    def teardown(self) -> None:  # noqa: B027
        """Free GPU memory."""
