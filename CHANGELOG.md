# Changelog

## [0.1.0] - unreleased

Working pipeline on real footage.

- Detector implemented: RT-DETR (Apache-2.0, hub-fetched, no API key) as the
  zero-setup default; AGPL YOLO refused inside `src/footy/` by design
- Tracker implemented: ByteTrack over player/goalkeeper/referee; ball excluded
  from motion-model association on purpose
- Team assignment implemented: grass-masked Lab-chroma torso statistics clustered
  per track; cluster centroids mapped to home/away by relative distance to the
  match config kit colours; gk/referee picked off as outliers
- Calibration: static homography end to end (solve, store, project ground points);
  per-frame mode abstains loudly pending a pitch keypoint model
- Ball trajectory: best-candidate selection, speed gating, bounded interpolation
- Orchestrator writes tracks.parquet (tracks_m contract), tracks_px.parquet,
  ball.parquet, and report.html; kinematics via analytics.physical
- Match config can override the camera profile; per-clip match configs
- Dependency floor fixed: numpy<2.0 (socceraction), supervision<0.31 (ByteTrack
  removal), floodlight dropped as unsatisfiable; CI installs the full test deps

Initial scaffold.

- Stage protocol and seven-stage run-time pipeline
- Config-driven camera, detector, tracker, and pitch profiles
- Schema contracts between stages
- xG with explicit local recalibration path
- Manual tagging bridge as an alternative to CV event detection
- Licence isolation: MIT core, GPL backends via subprocess only
