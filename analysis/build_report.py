"""Assemble the match report page: inject data and inline the map-view charts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CHARTS = OUT / "charts"


def svg(name: str) -> str:
    s = (CHARTS / f"{name}.svg").read_text(encoding="utf-8")
    s = re.sub(r"<\?xml.*?\?>", "", s, flags=re.S)
    s = re.sub(r"<!DOCTYPE.*?>", "", s, flags=re.S)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S).strip()
    s = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"', r"\1", s, count=1)
    s = re.sub(r'(<svg[^>]*?)\sheight="[^"]*"', r"\1", s, count=1)
    return s.replace("<svg ", '<svg role="img" ', 1)


def player_rows(pt: pd.DataFrame) -> str:
    out = []
    for _, r in pt.iterrows():
        kit = "home" if r["team"] == "home" else "away"
        out.append(
            f'<tr class="{kit}">'
            f'<td><span class="dot {kit}"></span><b>{r["player"]}</b></td>'
            f'<td><span class="band b-{r["role"].lower()}">{r["role"]}</span></td>'
            f'<td class="num">{r["minutes_on_camera"]:.2f}</td>'
            f'<td class="num">{r["distance_m"]:.0f}</td>'
            f'<td class="num">{r["m_per_min"]:.0f}</td>'
            f'<td class="num">{r["top_speed_ms"]:.2f}</td>'
            f'<td class="num">{r["high_speed_s"]:.1f}</td>'
            f'<td class="num">{int(r["sprints"])}</td>'
            f'<td class="num">{r["mean_x_m"]:.0f}, {r["mean_y_m"]:.0f}</td>'
            f"</tr>"
        )
    return "\n".join(out)


def main() -> int:
    clip = sys.argv[1] if len(sys.argv) > 1 else "clip0"
    s = json.loads((OUT / f"{clip}_summary.json").read_text())
    pt = pd.read_csv(OUT / f"{clip}_players.csv")
    tpl = (HERE / "report_template.html").read_text(encoding="utf-8")

    home_t = s["territory_pct"]["home"]
    away_t = s["territory_pct"]["away"]
    hs, as_ = s["shape"]["home"], s["shape"]["away"]
    hp = pt[pt.team == "home"]
    ap = pt[pt.team == "away"]

    repl = {
        "{{DURATION}}": f"{s['duration_s']:.1f}",
        "{{FRAMES}}": f"{s['frames']:,}",
        "{{PLAYERS}}": str(s["tracked_players"]),
        "{{ONCAM}}": f"{s['mean_players_on_camera']:.1f}",
        "{{POSS_HOME}}": f"{s['possession_pct'].get('home', 0):.0f}",
        "{{POSS_AWAY}}": f"{s['possession_pct'].get('away', 0):.0f}",
        "{{POSS_FRAMES}}": str(s.get("possession_frames", 0)),
        "{{H_ATT}}": f"{home_t['attacking']:.0f}",
        "{{H_MID}}": f"{home_t['middle']:.0f}",
        "{{H_DEF}}": f"{home_t['defensive']:.0f}",
        "{{A_ATT}}": f"{away_t['attacking']:.0f}",
        "{{A_MID}}": f"{away_t['middle']:.0f}",
        "{{A_DEF}}": f"{away_t['defensive']:.0f}",
        "{{H_WIDTH}}": f"{hs['mean_width_m']:.1f}",
        "{{A_WIDTH}}": f"{as_['mean_width_m']:.1f}",
        "{{H_DEPTH}}": f"{hs['mean_depth_m']:.1f}",
        "{{A_DEPTH}}": f"{as_['mean_depth_m']:.1f}",
        "{{H_CENT}}": f"{hs['mean_centroid_x_m']:.1f}",
        "{{A_CENT}}": f"{as_['mean_centroid_x_m']:.1f}",
        "{{TOPSPEED}}": f"{s['physical']['max_top_speed_ms']:.2f}",
        "{{MEDMPM}}": f"{s['physical']['median_m_per_min']:.0f}",
        "{{SPRINTS}}": str(s["physical"]["total_sprints"]),
        "{{NOISE}}": f"{s['noise_floor']['median_residual_m']:.2f}",
        "{{H_TOPSPEED}}": f"{hp.top_speed_ms.max():.2f}",
        "{{A_TOPSPEED}}": f"{ap.top_speed_ms.max():.2f}",
        "{{H_MPM}}": f"{hp.m_per_min.median():.0f}",
        "{{A_MPM}}": f"{ap.m_per_min.median():.0f}",
        "{{H_SPRINTS}}": str(int(hp.sprints.sum())),
        "{{A_SPRINTS}}": str(int(ap.sprints.sum())),
        "{{PLAYER_ROWS}}": player_rows(pt),
        "{{CHART_FORMATION}}": svg("formation"),
        "{{CHART_HEATMAP}}": svg("heatmap"),
        "{{CHART_BALL}}": svg("ball_path"),
        "{{CHART_CENTROID}}": svg("centroid"),
        "{{CHART_PHYSICAL}}": svg("physical"),
    }
    for k, v in repl.items():
        tpl = tpl.replace(k, v)

    left = re.findall(r"\{\{[A-Z_]+\}\}", tpl)
    if left:
        print("UNFILLED:", set(left))
    dst = OUT / "match_report.html"
    dst.write_text(tpl, encoding="utf-8")
    print("wrote", dst, f"({len(tpl):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
