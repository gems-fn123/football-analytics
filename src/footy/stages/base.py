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


class Stage(ABC):
    name: str = "stage"
    requires_gpu: bool = False

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.log = get_logger(f"footy.{self.name}")

    def setup(self) -> None:
        """Load weights. Called once before the first run()."""

    @abstractmethod
    def run(self, ctx: dict[str, Any]) -> StageResult:
        """ctx carries the outputs of earlier stages, keyed by stage name."""

    def teardown(self) -> None:
        """Free GPU memory."""
