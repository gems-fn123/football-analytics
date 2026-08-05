# Runbook

## Before the first match

1. Decide the camera profile. This is the single highest-leverage choice.
   Fixed wide beats broadcast by a wide margin.
2. Measure the pitch if it is non-standard. Edit `configs/pitch/standard_105x68.yaml`.
   Wrong dimensions bias every distance and xG number.
3. For a fixed camera, solve the homography once:
   `python scripts/calibrate_fixed_camera.py --video data/raw/sample.mp4`
4. Fetch weights: `make weights`

## Per match

1. Fill in the squad list: copy `configs/match/example_match.yaml`, edit shirt numbers.
   Five minutes of typing. Skip it and you get "home #7" instead of names.
2. Smoke test 300 frames:
   `footy run --video <file> --match configs/match/<match>.yaml --max-frames 300`
3. Eyeball `data/processed/<match_id>/tracks.parquet`. Check that both teams appear,
   that x_m and y_m fall inside 0-105 and 0-68, and that track_id counts look sane.
4. Full run: `bash scripts/run_match.sh <file> configs/match/<match>.yaml`

Budget roughly one hour per 90 minutes of footage on a cloud GPU. CPU-only is not
practical for a full match.

## Getting events

Two routes.

**Auto.** Set `stages.events: true`. Fast, coarse. Good for shots, goals, corners.
Weak on the long tail of duels, pressures, and carries.

**Manual, recommended.** Tag in LongoMatch or Kinovea while watching. Export. Then:

```bash
python scripts/export_tagging_template.py --make-template tagging.csv
# tag the match, export, then
python scripts/export_tagging_template.py --from-longomatch tags.xml --out data/interim/events.parquet
```

Budget roughly 2-3x match duration for a first pass by a single tagger. Slower than the
CV route, and far more accurate. For event data specifically, this is how you get closest
to Opta depth.

## Recalibrating xG

Collect around 250 local shots with known outcomes, then:

```bash
footy fit-xg --shots data/processed/local_shots.parquet --out models/weights/xg_local.pkl
```

Until you do this, xG values inherit elite-league conversion rates and will run high.

## Sanity checks that catch most bugs

- Total distance per player per 90: expect 9-12 km for outfield. Wildly off means the
  homography or the pitch dimensions are wrong.
- Top speed: above 11 m/s means tracking noise, not a sprinter.
- Possession share: should sum to roughly 100 once loose-ball frames are excluded.
- Identity coverage: if under 50 percent, prefer shirt-number-free team-level metrics.
