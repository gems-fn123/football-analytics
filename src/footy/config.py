"""Config loading. Plain YAML, no framework, resolves nested config file paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

NESTED_KEYS = ("camera", "detector", "tracker", "pitch")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Config:
    """Resolved run-time config. Nested config paths are expanded in place."""

    raw: dict[str, Any]
    match: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, pipeline_path: str | Path, match_path: str | Path | None = None) -> Config:
        raw = load_yaml(pipeline_path)
        match = load_yaml(match_path) if match_path else {}
        # A YAML section left empty ("io:" with no children) loads as None, and every
        # consumer expects a dict. Normalise once here rather than `or {}` everywhere.
        raw = {k: ({} if v is None else v) for k, v in raw.items()}
        match = {k: ({} if v is None else v) for k, v in match.items()}
        # The camera (and in principle any nested profile) belongs to the venue and
        # recording, so the per-match file may override the pipeline default:
        #   camera: configs/camera/broadcast.yaml
        for key in NESTED_KEYS:
            override = match.get(key)
            value = override if isinstance(override, str) else raw.get(key)
            if isinstance(value, str):
                raw[key] = load_yaml(value)
        return cls(raw=raw, match=match)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def match_id(self) -> str:
        return self.match.get("match_id") or self.raw.get("match_id", "unnamed_match")

    def squad_map(self) -> dict[tuple[str, int], str]:
        """(team_key, shirt_number) -> player name. Empty if no squad list given."""
        out: dict[tuple[str, int], str] = {}
        for side in ("home", "away"):
            squad = (self.match.get(side) or {}).get("squad") or {}
            for shirt, info in squad.items():
                out[(side, int(shirt))] = info.get("name", f"{side} #{shirt}")
        return out
