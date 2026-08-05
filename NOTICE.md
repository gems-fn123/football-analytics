# Third-party licence matrix

Read this before distributing anything built on this repo.

## Safe by default (permissive)

| Component | Licence | Used for |
|---|---|---|
| roboflow/sports | MIT | detection, team clustering, radar |
| supervision | MIT | ByteTrack, annotation, slicing |
| RF-DETR | Apache-2.0 | detector backbone (default) |
| kloppy | BSD-3 | data standardisation |
| socceraction | MIT | SPADL, xT, VAEP |
| mplsoccer | MIT | pitch plots |
| floodlight | MIT | physical metrics |
| SAM 2 | Apache-2.0 | segmentation tracking |

## Copyleft - review before commercial use

| Component | Licence | Consequence |
|---|---|---|
| Ultralytics YOLOv8 / v11 | AGPL-3.0 | Network use triggers source disclosure. Buy a commercial licence or use RF-DETR. |
| SoccerNet/sn-gamestate | GPL-3.0 | Derivative works must be GPL. Keep as an external process, not a linked import. |
| PnLCalib | GPL-2.0 | Same. Isolate behind a subprocess boundary. |
| LongoMatch | GPLv2 | Standalone GUI. Using its exported files is fine. |

## Unclear or absent licence - do not vendor

| Component | Status |
|---|---|
| abdullahtarek/football_analysis | No LICENSE file. Treat as all-rights-reserved. Reference only, do not copy code. |
| No-Bells-Just-Whistles | LICENSE file present but not auto-classified. Open and read it before use. |

## Data

| Dataset | Terms |
|---|---|
| StatsBomb Open Data | Custom, requires attribution "Data provided by StatsBomb" |
| Wyscout / Pappalardo | CC BY 4.0 |
| PFF FC 2022 WC | Free on request, check the form terms |
| SkillCorner open | Open with credit |
| SoccerNet | Research use, NDA password required |

## Isolation rule

Anything GPL or AGPL is invoked as a **subprocess** via `scripts/`, never imported into
`src/footy/`. This keeps the core package MIT-clean.
