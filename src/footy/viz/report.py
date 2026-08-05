"""Single-file HTML match report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>{match_id} report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}}
.caveat{{background:#fff8e1;border-left:3px solid #f0a500;padding:10px 14px;font-size:14px}}
</style>
<h1>{match_id}</h1>
<p class="caveat">Generated from video by an open-source pipeline. Identity resolution is
approximately {identity_pct}% complete and event coverage is partial. Treat these numbers as
indicative, not as provider-grade data.</p>
{sections}
"""


def build_report(
    match_id: str,
    tables: dict[str, pd.DataFrame],
    out_path: str | Path,
    identity_pct: float = 0.0,
) -> Path:
    sections = "".join(
        f"<h2>{name}</h2>{df.head(50).to_html(index=False, border=0)}"
        for name, df in tables.items()
    )
    html = TEMPLATE.format(
        match_id=match_id, sections=sections, identity_pct=round(identity_pct * 100)
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
