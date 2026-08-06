# Pre-registration — analytic success criteria

**Status: FROZEN before any pipeline run.**
Written 2026-08-05. No criterion in this document may be edited after results exist.
Failures get reported as failures. Nothing here is tuned to an observed output.

---

## 0. Why this document exists

It is trivially easy to run a CV pipeline, look at whatever comes out, and write a
report describing that output as a success. This document fixes the bar first, so
the results are scored against a standard set in ignorance of them.

Everything below is either:
- **C-n** — a criterion with a numeric pass/fail threshold, or
- **P-n** — a falsifiable *prediction* about what the pipeline will do. A wrong
  prediction is a finding, not an embarrassment; it means the system does not behave
  the way its own documentation claims.

---

## 1. Material under test

Three iStock/Getty stock clips, treated per instruction as *one candidate real camera
angle*. All 768×432, 25 fps.

| Clip | File | Frames | Duration | Content (from config + visual check) |
|---|---|---|---|---|
| clip0 | `smoke_clip.mp4` | 807 | 32.3 s | Kickoff, blue vs white, yellow referee |
| clip1 | `smoke_clip (1).mp4` | 880 | 35.2 s | Open play, red vs blue, green GK |
| clip2 | `smoke_clip (2).mp4` | 364 | 14.6 s | Attack on right goal, red vs cyan |

**Measured camera behaviour** (phase correlation, before any pipeline run):

| Clip | Cumulative pan (half-res px) | Peak excursion | Hard cuts |
|---|---|---|---|
| clip0 | x +149.5, y −55.3 | 243.1 | 0 |
| clip1 | x −173.1, y +1.6 | 176.1 | 0 |
| clip2 | x +55.7, y −53.0 | 55.7 | 0 |

All three cameras pan continuously. None is locked off. This is a measurement, not an
assumption, and it drives P-1 and P-2 below.

---

## 2. Known material hazards (declared up front)

- **H-1 Crowd.** The stands are densely occupied. The detector is COCO RT-DETR, whose
  `person` class has no concept of "on the pitch". Crowd bleed is the single largest
  threat to every person-derived number.
- **H-2 Watermark.** A large iStock/Getty watermark sits across frame centre, exactly
  where play happens. It occludes players.
- **H-3 Resolution.** 768×432. Players are roughly 20–45 px tall. Shirt numbers are
  physically unreadable at this scale.
- **H-4 Synthetic source.** Rendered/CGI footage, not real broadcast. Detector domain
  fit is unknown; results may not transfer to real footage in either direction.

---

## 3. Criteria

### A. Ingest

- **C-A1** All three clips decode; frames processed equals the probe frame count
  (allowing the documented ±1 for the final frame).

### B. Detection

Anchor: the repo's own `configs/camera/broadcast.yaml` declares
`players_visible: 15` as typical for this camera class. That is the pre-existing,
independent expectation, so criteria are set around it rather than around anything
this run produces.

- **C-B1 (recall floor)** Median persons detected per frame ≥ **10**.
  Rationale: the repo expects ~15 visible; finding fewer than two-thirds of them makes
  team shape unreliable.
- **C-B2 (false-positive ceiling)** Median persons detected per frame ≤ **28**.
  Rationale: the hard physical maximum on a pitch is 22 players + 3 officials = 25.
  Sustained counts above 28 can only come from off-pitch detections.
- **C-B3 (crowd contamination test, decisive)** Median persons/frame ≤ **35**.
  If exceeded, H-1 is confirmed, and **every person-derived metric in this report is
  declared invalid** until a spatial gate is added. This criterion outranks all others
  in section C and D — they are not scored as meaningful if C-B3 fails.
- **C-B4 (ball presence)** Ball detected in ≥ **20 %** of processed frames.
  Rationale: a fast, small, frequently occluded object; 20 % is a deliberately low bar
  that still permits a usable gated trajectory.
- **C-B5 (dropout)** Frames with zero person detections ≤ **5 %**.

### C. Tracking

- **C-C1 (continuity)** Median track lifetime ≥ **25 frames** (1.0 s at 25 fps).
- **C-C2 (identity churn)** Distinct `track_id` count ≤ **5 ×** median persons/frame.
  Rationale: a perfect tracker on a 15-player scene yields ~15–25 ids; 5× is a
  generous ceiling that still catches runaway fragmentation.

### D. Team assignment

This is the largest new feature in the pushed commit (253 changed lines) and the part
most worth testing.

- **C-D1 (coverage)** ≥ **60 %** of sampled tracks receive a non-null team.
- **C-D2 (balance)** min(home, away) / max(home, away) ≥ **0.40** over assigned field
  tracks. Rationale: both sides are on screen in all three clips; a wildly lopsided
  split means the clustering collapsed rather than separated.
- **C-D3 (correctness)** The home/away label mapping agrees with the kit colours
  visible in the footage, checked by eye against named frames and recorded in the
  results doc. Qualitative, but stated in advance and reported either way.

### E. Ball trajectory

- **C-E1** `ball.parquet` is non-empty.
- **C-E2 (honesty of interpolation)** Interpolated rows ≤ **50 %** of ball rows.
  Rationale: past half, the "trajectory" is mostly synthesised rather than observed.

### F. Calibration and the metric chain

- **P-1 (prediction: metres unavailable)** All three clips are configured to
  `broadcast.yaml` → `homography.mode: per_frame`, and `stages/calibrate.py` abstains
  in that mode by design. **Prediction: `x_m` and `y_m` are 100 % NaN in
  `tracks.parquet`, and the run logs a loud warning.** Any non-NaN metre value would
  falsify this and indicate a defect.
- **P-2 (prediction: static homography is not a valid rescue)** Given the measured pan
  in §1, substituting a single static homography must produce position error that grows
  with camera excursion. **Criterion C-F1:** a static homography is acceptable only if
  induced positional error stays < **2.0 m** across the clip. That threshold is not
  invented here — it is the repo's own warning bound in `cli.py`
  (`if err > 2.0: log.warning("reprojection error above 2 m")`).
  **Prediction: FAIL on all three clips.**
- **C-F2** If C-F1 fails, metres are reported as *unavailable*, and **no distance,
  speed, or physical metric is published for these clips**. Fabricating them is the
  specific failure this whole document exists to prevent.

### G. Physical metrics

- **P-3 (prediction)** With metres NaN, `analytics.physical.summarise_player` returns
  an empty frame and the report omits the "Physical per player" table. Predicted
  consequence: **no distance covered, no top speed, no sprint counts** for any clip.

### H. Deliverable integrity

- **C-H1** Every figure in the analyst report traces to a named pipeline output file
  or an explicitly shown computation. No number appears without provenance.
- **C-H2** Failed criteria appear in the final report, with the failure stated plainly.
- **C-H3** No watermarked stock frame is embedded in any published deliverable
  (licensing — the source footage is a watermarked preview, not a licensed asset).

---

## 4. Decision rules

Stated before results, so the verdict is mechanical rather than editorial:

| Verdict | Requires |
|---|---|
| **Usable for tactical / shape analysis** | C-B3 passes, and C-B1, C-C1, C-D1, C-D2 all pass |
| **Usable for physical analysis** | C-F1 passes (metres trustworthy) |
| **Usable for event analysis** | C-E1, C-E2 pass *and* metres available |
| **Smoke-test only** | anything less |

Expected verdict on the strength of §1 and §2 alone: **smoke-test to tactical at best,
physical analysis blocked.** Recorded here so the final report cannot quietly claim more.

---

## 5. Ground-truth protocol

To measure detection recall independently of the detector, persons visibly on the
pitch are counted by hand on one frame per clip, from 2× upscaled stills, *before* the
pipeline is run. Counts, and their uncertainty, are recorded in `01_ground_truth.md`.
Crowd, staff, and anyone outside the field of play are excluded — that boundary is
precisely what C-B3 tests.
