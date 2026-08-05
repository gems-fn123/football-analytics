# Third-party licence matrix

Read this before distributing anything built on this repo.

## Safe by default (permissive)

| Component | Licence | Used for |
|---|---|---|
| roboflow/sports | MIT | detection, team clustering, radar |
| supervision | MIT | ByteTrack, annotation, slicing |
| RF-DETR | Apache-2.0 | detector backbone (once a football fine-tune exists) |
| RT-DETR (`PekingU/rtdetr_r18vd`) | Apache-2.0 | detector backbone, current default. Model *and* weights are Apache-2.0, and the hub download needs no API key. |
| transformers | Apache-2.0 | RT-DETR loading and post-processing |
| kloppy | BSD-3 | data standardisation |
| socceraction | MIT | SPADL, xT, VAEP |
| mplsoccer | MIT | pitch plots |
| SAM 2 | Apache-2.0 | segmentation tracking |

### Permissively licensed but not usable here

Two packages are licence-clean yet excluded on dependency grounds, not legal ones:

| Component | Licence | Why it is out |
|---|---|---|
| floodlight | MIT | No version satisfies `pandas>=2.1` alongside socceraction: <0.5 needs `pandas<2.0`, >=0.5 needs `numpy>=2.1`. |
| trackers (`ByteTrackTracker`) | Apache-2.0 | Requires `numpy>=2.0.2`; socceraction caps `numpy<2.0`. This is why `supervision` is pinned `<0.31`, where `sv.ByteTrack` is removed. |

Both become available the day socceraction supports numpy 2.x. Until then, adopting either
means giving up SPADL, xT, and VAEP.

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
