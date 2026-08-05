"""Pipeline stages. Each one is a Stage: a config in, a DataFrame out."""

from footy.stages.base import Stage, StageResult

__all__ = ["Stage", "StageResult"]
