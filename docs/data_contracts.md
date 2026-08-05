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

detections plus `track_id` (int64). `-1` means untracked.

## tracks_m

| column | type | note |
|---|---|---|
| frame | int64 | |
| t_s | float32 | seconds from video start |
| track_id | int64 | |
| team | string | home, away, referee, ball |
| shirt | Int16 | **nullable** - null means OCR abstained |
| player | string | **nullable** - resolved from the squad list |
| x_m y_m | float32 | pitch metres, origin bottom-left |
| speed_ms | float32 | added by analytics.physical |
| conf | float32 | detection confidence, carried through |

Nullability is deliberate. A null shirt is honest; a guessed shirt corrupts every
per-player metric downstream.

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
