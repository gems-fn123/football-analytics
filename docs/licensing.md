# Licensing

See `NOTICE.md` for the full matrix. The short version:

- Own code: MIT.
- Default detector backend: Apache-2.0. Safe.
- Ultralytics YOLO is AGPL-3.0. Opt-in only, via `configs/detector/yolo.yaml`.
  If you serve this over a network with YOLO enabled, AGPL source-disclosure applies
  to your whole service unless you hold a commercial licence.
- GPL repos (sn-gamestate, PnLCalib) are invoked as subprocesses only.
- Datasets carry their own terms and often require attribution. StatsBomb open data
  requires the line "Data provided by StatsBomb" wherever it is used.

## Rule for contributors

Never `import` a GPL or AGPL package inside `src/footy/`. Wrap it in `scripts/` and
call it with `subprocess`. The CI does not enforce this; review does.
