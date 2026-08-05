# Architecture

## Two clocks

The repo separates two things that are easy to conflate.

**Run time** happens once per match. Input: a video file and a squad list. Output: tables.
No network access, no provider API, no dataset lookup. This path is league-agnostic.

**Train time** happens rarely, offline. Input: public datasets. Output: weights files.
Once a weights file exists it is a frozen artefact; the run-time path treats it as a
constant.

The only league-dependent artefact is the xG model, because conversion rates vary by
competition. Geometry does not.

## Stage contract

Every stage subclasses `footy.stages.base.Stage` and implements `run(ctx) -> StageResult`.
`ctx` is a dict carrying prior results keyed by stage name, plus `video` and `config`.

This means stages can be:
- skipped via `configs/pipeline.yaml` under `stages:`
- reordered, as long as declared inputs still exist in `ctx`
- swapped for a different backend by pointing the config at another yaml

Skipping has structure. `detect` feeds everything, and `track` feeds everything
after it - disabling either fails fast with a message naming the missing stage.
The enrichment stages (`team`, `identity`) and `calibrate`, `ball`, `events` are
individually skippable: downstream stages fill the columns a skipped enricher
would have added with nulls, which is exactly what "that stage did not run" means.

## Data flow

```
video file
  -> io.video.VideoSource            frames
  -> stages.detect                   boxes per frame
  -> stages.track                    boxes with track_id
  -> stages.team                     + team label
  -> stages.identity                 + shirt, player
  -> stages.calibrate                + x_m, y_m
  -> stages.ball                     ball trajectory
  -> stages.events                   SPADL-shaped actions
  -> analytics.*                     metrics
  -> viz.*                           plots and report
```

## GPL isolation

Anything GPL or AGPL runs as a subprocess launched from `scripts/`, never imported into
`src/footy/`. This keeps the core package MIT-clean and distributable. If you add a GPL
backend, add a script wrapper, not an import.

## Where it breaks

| Stage | Failure mode | Mitigation in code |
|---|---|---|
| track | identity switch on crossings | re-ID backend, lost_track_buffer |
| identity | ~1 in 5 shirts misread | majority vote, confidence floor, abstain to null |
| calibrate | drift on moving cameras | per-frame solve, RANSAC |
| ball | gaps and false positives | speed gate, bounded interpolation |
| events | coarse taxonomy | manual route via scripts/export_tagging_template.py |
