# football-analytics

Open-source pipeline that turns a match video recording into tracking data, events, and analytics.

Scope and honest ceiling: this reconstructs a *partial, noisier* version of what Opta or StatsBomb sell.
Expect useful team-level tactics, physical metrics, and a calibrated xG model. Do not expect
1,600+ hand-verified events per match or 25 Hz identity-stable tracking for all 22 players.

## Run time vs train time

The per-match run is **league-agnostic**. It reads a video file and a squad list. It does not
query StatsBomb, Wyscout, SoccerNet, or any provider API.

Public datasets appear only upstream of training. Two things carry a league assumption:

| Thing | League dependent? | Action |
|---|---|---|
| Detector / tracker / calibration weights | No | Use as is |
| xG coefficients | Yes | Recalibrate on ~250 local shots |
| Detector domain fit (resolution, kit style, camera height) | Sort of | Fine-tune on your own frames if accuracy drops |

## Quick start

```bash
make setup
make weights
footy run --video data/raw/match.mp4 --config configs/pipeline.yaml --match configs/match/example_match.yaml
```

Outputs land in `data/processed/<match_id>/`:

```
tracks.parquet      frame, track_id, team, shirt, x_m, y_m
events.parquet      SPADL-compatible action table
report.html         plots and summary
```

## Pipeline stages

| Stage | Module | Needs GPU | Fragile? |
|---|---|---|---|
| 0 ingest | `footy.io.video` | no | no |
| 1 detect | `footy.stages.detect` | yes | no |
| 2 track | `footy.stages.track` | optional | ID switches |
| 3 team | `footy.stages.team` | optional | similar kits |
| 4 identity | `footy.stages.identity` | yes | **yes, ~0.78 acc** |
| 5 calibrate | `footy.stages.calibrate` | yes | moving cameras |
| 6 ball | `footy.stages.ball` | yes | **weakest link** |
| 7 events | `footy.stages.events` | yes | coarse vs Opta |

Stages 6 and 7 can be bypassed entirely by hand-tagging in LongoMatch or Kinovea and importing
via `scripts/export_tagging_template.py`. For event data specifically, that route is faster and
more accurate than the CV route.

## Camera profile drives everything

| Profile | Config | What you get |
|---|---|---|
| Fixed wide / tactical | `configs/camera/fixed_wide.yaml` | Best case. Calibrate once, all 22 players in frame. |
| Broadcast | `configs/camera/broadcast.yaml` | On-ball only, ~14-16 players visible, per-frame homography. |
| Phone | `configs/camera/phone.yaml` | Short sequences only. |

## Licensing warning

Own code is MIT. Several optional backends are AGPL-3.0 or GPL. See `NOTICE.md` before any
commercial or closed-source use. The default detector backend is chosen to stay permissive.

## Docs

- `docs/architecture.md` - stage contracts and data flow
- `docs/data_contracts.md` - column schemas between stages
- `docs/runbook.md` - operating a full match end to end
- `docs/licensing.md` - dependency licence matrix
