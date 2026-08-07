# Pre-registration — metric chain (metres, distances, events)

**Status: FROZEN before the homography tracker was run.**
Written 2026-08-07. Extends `00_preregistration.md`, which froze the criteria for
detection, tracking, teams and ball. No criterion here may be edited after results exist.

---

## 0. What changed since 00_preregistration.md

`00_preregistration.md` predicted (P-1) that metres would be 100 % NaN, and (P-2/C-F1)
that substituting a single static homography would fail. Both were confirmed:

| clip | static-H median error | breaches 2 m at | C-F1 |
|---|---|---|---|
| clip0 | 10.75 m | 1.2 s | FAIL |
| clip1 | 11.37 m | 0.8 s | FAIL |
| clip2 | 10.01 m | 0.8 s | FAIL |

That result is **not** evidence that metres are unobtainable. It measures the cost of
holding one homography *fixed* while the camera moves. The quantity
`calibration_feasibility.py` reports as "error" is the camera motion itself, measured
frame by frame in the image. If the homography is *tracked* rather than held fixed, that
motion is compensated instead of accumulated.

This document fixes the bar for that claim before testing it.

## 1. Method under test

**Anchored homography tracking.** For clip0 only, since it is the only clip with a
verified pitch anchor (`analysis/calib/clip0_reference.json`, 1.9 px median).

1. `H_pitch→0` is the verified reference at frame 0.
2. Image-to-image homographies are estimated between frames by feature matching on
   *static scene content only*. Two things must be masked or the estimate is corrupted:
   - **players**, which move independently of the camera, and
   - **the burnt-in watermark**, which is fixed to the *image* rather than the ground.
     This is the more dangerous of the two: watermark features match at zero
     displacement across every frame, so RANSAC can select them as a consensus set and
     return H ≈ identity, silently reporting "the camera did not move".
3. Frames are matched to **keyframes**, not chained one-by-one; a new keyframe is
   started when match quality drops. Chaining only keyframes keeps the number of
   composed homographies small, since composition multiplies error.
4. `H_pitch→t = H_keyframe→t · (chain of keyframe hops) · H_pitch→0`.

## 2. Criteria

Errors are expressed in **pitch metres**, obtained by unprojecting image points through
the homographies being compared — not via a player-height proxy. The 2.0 m threshold is
not invented here: it is the repo's own reprojection warning bound in `cli.py`.

### A. Homography tracking

- **C-K1 (coverage)** ≥ **95 %** of frames receive a homography.
  Rationale: an unsolved frame means null metres for every player in it.
- **C-K2 (drift, decisive)** For frames where *direct* frame-0 matching independently
  succeeds with ≥ 100 RANSAC inliers, the tracked homography must agree with that direct
  solve to < **2.0 m** median over a pitch grid.
  This is the central test: it measures accumulated composition error against an
  independent estimate that involved no chaining at all. **If C-K2 fails, metres are
  reported as unavailable and no distance or speed is published** — the same rule as
  C-F2 in `00_preregistration.md`.
- **C-K3 (no silent identity)** The tracked homography must reproduce the camera motion
  already measured independently in `calibration_feasibility.json`: cumulative image
  displacement must reach ≥ 100 px by end of clip0. Guards specifically against the
  watermark failure mode in §1.2, which would otherwise pass C-K2 trivially by
  reporting no motion at all.
- **C-K4 (on-pitch plausibility)** ≥ **90 %** of tracked player detections project
  inside the pitch with a 3 m margin. This is the pipeline's own `fraction_on_pitch`
  check, and it catches a globally misplaced pitch that C-K2 cannot.

### B. Physical metrics — only scored if C-K2 passes

Thresholds are the runbook's own sanity bounds (`docs/runbook.md`), chosen because they
predate this work.

- **C-P1 (top speed)** ≤ **1 %** of per-player frame-to-frame speeds exceed **11 m/s**.
  The runbook states "above 11 m/s means tracking noise, not a sprinter".
- **C-P2 (distance rate)** Median player distance rate, extrapolated to 90 minutes,
  falls in **9–12 km**. Stated with the caveat that these are 15–35 s clips: the
  extrapolation is a plausibility check on the *rate*, not a claim about match volume,
  and will be reported as a rate.
- **C-P3 (no fabricated coverage)** Any player-frame whose homography is missing, or
  whose projection lands outside the C-K4 margin, is **null**, never interpolated.

### C. Tracking quality (re-scoring an already-failed criterion)

`00_preregistration.md` C-C2 failed on clip0 (87 ids vs cap 75) and clip1 (179 vs 92).
OSNet appearance stitching now exists in the repo, disabled by default.

- **C-S1 (fragmentation)** With stitching enabled, distinct `track_id` count ≤ **5 ×**
  median persons/frame — i.e. C-C2 as originally written, re-run.
- **C-S2 (no false merges, decisive)** A merge that fuses two different players is worse
  than fragmentation, per the repo's own stated reasoning. Merged tracks must not
  overlap in time, and **no merged track may contain two detections in the same frame**.
  A violation is proof of a false merge. If C-S2 fails, stitching stays off.

## 3. Predictions

Recorded so they can be wrong.

- **P-4** C-K2 will pass on clip0. Composition error over ~20 keyframes should stay well
  under a metre, since each hop is a small-baseline match with hundreds of inliers.
- **P-5** C-P1 will **fail** on the first attempt. Fragmented tracks produce identity
  switches, and an identity switch teleports a "player" across the pitch in one frame,
  which reads as an impossible speed. This predicts the physical metrics need C-S1
  fixed first, not just metres.
- **P-6** clip1 and clip2 will remain unavailable for metres regardless of C-K2, because
  neither has a verified pitch anchor. Tracking propagates an anchor; it cannot create
  one.

## 4. Scope limit stated up front

Passing every criterion here yields **metres, distances and speeds for clip0 only**.
Events additionally require the ball, whose own criterion (C-B4) already failed on
clip2. Nothing in this document licenses publishing per-player physical metrics for
clip1 or clip2.
