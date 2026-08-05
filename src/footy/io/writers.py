"""Table output. Parquet by default, CSV when a human needs to eyeball it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, path: str | Path, also_csv: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    if also_csv:
        df.to_csv(path.with_suffix(".csv"), index=False)
    return path
