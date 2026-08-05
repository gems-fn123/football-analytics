# Data contracts

Column schemas live in `src/footy/schemas.py` and are enforced at each stage boundary.

## detections

| column | type | note |
|---|---|---|
| frame | int64 | source frame index, not sequential if stride > 1 |
| det_id | int64 | unique within frame |
| cls | string | player, goalkeeper, referee, ball |
| conf | float32 | |
| x1 y1 x2 y2 | float32 | pixel coordinates |

## tracks_px

detections plus `track_id` (int64). `-1` means untracked; ball rows keep `-1` by
design because the ball is handled by its own stage, not the motion-model tracker.

The team and identity stages append `team`, `shirt`, and `player` to this shape.
The enriched pixel table is written as `tracks_px.parquet` - it is the ground truth
for debugging, and the only tracks output when the camera is uncalibrated.

## tracks_m

| column | type | note |
|---|---|---|
| frame | int64 | |
| t_s | float32 | seconds from video start, added by stages.calibrate |
| track_id | int64 | |
| team | string | home, away, referee, ball; **nullable** - null means unassigned |
| shirt | Int16 | **nullable** - null means OCR abstained |
| player | string | **nullable** - resolved from the squad list |
| x_m y_m | float32 | pitch metres, origin bottom-left; NaN when uncalibrated |
| speed_ms | float32 | added by analytics.physical in the orchestrator |
| conf | float32 | detection confidence, carried through |

Nullability is deliberate. A null shirt is honest; a guessed shirt corrupts every
per-player metric downstream. The same goes for metres: an uncalibrated camera
produces NaN positions, never invented ones.

## ball

Separate from tracks_m: the ball needs its own gating and interpolation, and the
ground-plane homography misplaces an airborne ball, so metres here are approximate.

| column | type | note |
|---|---|---|
| frame | int64 | |
| t_s | float32 | |
| x_px y_px | float32 | pixel centre of the ball box |
| x_m y_m | float32 | via homography; NaN when uncalibrated |
| conf | float32 | NaN on interpolated rows |
| interpolated | bool | true = filled linearly across a gap <= max_gap_frames |

Gaps longer than `max_gap_frames` have no rows at all. Null beats invented.

## events

SPADL-compatible subset. See `analytics.spadl.SPADL_TYPE_MAP` for the type vocabulary.

| column | type |
|---|---|
| match_id | string |
| period | int8 |
| t_s | float32 |
| team | string |
| player | string |
| type_name | string |
| start_x start_y end_x end_y | float32 |
| result_name | string |
| bodypart_name | string |

Coordinates are pitch metres, same frame as tracks_m. socceraction expects a 105x68
pitch, so no rescaling is needed.
