# Results — metric chain

Scored against `02_preregistration_metrics.md`, which was frozen before any of this ran.
Criteria that failed are reported as failed.

## Headline

**Calibration is solved. It is no longer the thing blocking metres.**
**Tracking is now the binding constraint, and it was not previously identified as one.**

That reversal is the main finding. Before this work the assumption — mine included, and
the reason tasks 17 and 20 existed — was that registration accuracy was the obstacle. It
is not.

## A. Homography tracking

| criterion | bar | result | |
|---|---|---|---|
| C-K1 coverage | ≥ 95 % | **100 %** (807/807 frames) | PASS |
| C-K2 drift | < 2.0 m | **0.192 m** median, p90 0.467, max 0.712 | PASS |
| C-K3 real motion | ≥ 100 px | **1934 px** cumulative | PASS |
| C-K4 on-pitch | ≥ 90 % | **70.8 %** | **FAIL** |

Against the static homography this replaces, measured by the same pipeline:

| | median positional error |
|---|---|
| single static homography (`calibration_feasibility.json`) | 10.75 m |
| tracked homography (this work) | **0.192 m** |

A 56× improvement, and 10 keyframes over 807 frames with **no accumulation trend**
(error vs frame index correlates +0.232; pairs crossing a keyframe boundary sit at
0.239 m against 0.170 m within one keyframe, so composition is not the limiting term).

### C-K2 was measured wrongly the first time

The first run reported 0.000 m drift at frames 25–100. That is not a good result, it is
a broken test: while the keyframe is still frame 0, the "tracked" and the "direct"
homography are *the same computation*, so the check compared a number with itself. Only
frame 300 (1.384 m) was independent.

Replaced with three-way consistency — for a frame pair (a, b), a freshly matched
`H_a→b` must satisfy `tracked H_0→b = H_a→b · tracked H_0→a`. That involves no shared
computation and works anywhere in the clip. The 0.192 m figure is from 26 such pairs.

### C-K4 failed, and the cause is the detector, not the geometry

82.6 % of off-pitch projections fall outside the **goal lines**, sit higher in the image
(median 174 px vs 216 px for on-pitch) and carry lower confidence (0.729 vs 0.818).
Those are spectators near the horizon, where a correct homography legitimately maps a
few pixels to enormous ground distances — the declared **H-1** hazard from
`00_preregistration.md`, not a displaced pitch. A displaced pitch would move every
detection coherently and would not sort by image height.

Consequence: `C-K4` as written measures crowd contamination rather than registration.
It still fails, and the spatial gate (C-P3) is what handles it downstream.

## B. Physical metrics

| criterion | bar | result | |
|---|---|---|---|
| C-P1 speeds over 11 m/s | ≤ 1 % | **1.68 %** | **FAIL** |
| C-P2 distance rate | 9–12 km/90 | **23.4** | **FAIL** |
| C-P3 gated rows null | — | 71.3 % kept, rest null | PASS |

**P-5 was correct**: C-P1 failed on the first attempt. Two fixes moved it a long way,
and both are derived rather than tuned:

1. **The velocity estimator was wrong for this noise level.** `add_kinematics` smooths
   positions then differences *adjacent* frames, which divides position noise by
   dt = 0.04 s. Measured position noise is σ ≈ 0.21 m, so single-frame differencing
   carries σ_v = 7.4 m/s of pure noise — which is why 20.1 % of raw per-frame steps were
   already physically impossible, and why the "median speed" was mostly noise.
   Replaced with a windowed OLS fit whose length is *derived* from σ (not swept):
   σ_v = σ/(dt·√(N(N²−1)/12)), requiring σ_v ≤ 0.5 m/s gives N ≥ 12.
2. **Tracks that teleport are not one player.** Split wherever a step exceeds
   `12 m/s·dt + 3σ√2` — impossible by more than 3σ, so noise never triggers it.
   87 tracks → 251 segments.

| | speeds over 11 m/s | max speed |
|---|---|---|
| per-frame differencing (repo default) | 7.73 % | 102.8 m/s |
| + derived-window estimator | 4.47 % | 107.3 m/s |
| + physics-based track splitting | **1.68 %** | **15.7 m/s** |

Still short of the 1 % bar, and **C-P2 is not close** — 23.4 km/90 against a 9–12 bar.

### Why C-P2 cannot be met as written

Median segment life after splitting is **1.6 s**. Extrapolating 1.6 s to 90 minutes
amplifies any bias enormously, and short surviving segments are precisely the moving
ones. The criterion is not measurable at this track length; that is a statement about
the criterion's applicability to 32 s of active play, not a result to be worked around.
It is recorded as a failure either way.

## C. What is actually limiting accuracy now

Two residual causes, separated by measurement rather than assumed.

**1. Tracking quality — dominant.** 46 % of tracks contained physically impossible
motion before splitting. C-C2 already failed in `00_preregistration.md` (87 ids vs a cap
of 75). With median segment life 1.6 s, per-player distance is not a measurable quantity
regardless of how good the geometry is.

**2. Ground-plane registration — real but modest.** The image registration locks onto
the **stands**, not the pitch: grass is nearly textureless, yielding 379 SIFT features
against 1494 off-grass (80 % of features), and grass-only matching fails outright at
every frame tested. A homography fitted to the stands describes the ground plane only if
the camera rotates purely about its optical centre.

Tested for, rather than assumed, by the common-mode signature — a registration error
makes *every* player drift together in proportion to camera speed:

| | common-mode player speed |
|---|---|
| camera nearly still (slowest 25 % of frames) | 2.91 m/s |
| camera panning fast (fastest 25 %) | 4.32 m/s |

A 1.49× effect, correlation with pan rate only +0.194. So the effect is **present and
directional, but not the dominant term** — most of the common mode survives a stationary
camera and is genuine collective play plus residual noise. Recorded at that strength
deliberately: the first reading of this test suggested a 2.2× anomaly, and controlling
for camera pan cut it down.

## D. What this means for the approach

The teammate's per-frame keypoint calibration (`src/footy/calib/`) constrains the
homography with **ground-plane** features — pitch keypoints — which is exactly the term
image registration cannot supply here, since the ground is textureless and the stands
are not on it. This measurement is independent support for that design over the
image-registration route, and it is why task 20 (cone–conic, a better *single-frame*
fit) is parked: single-frame accuracy was never the limiting quantity.

Blocking that path: the committed channel→pitch grid covers 35 of 57 channels and misses
every channel our footage fires on. `analysis/derive_grid_clip0.py` recovered 3 more by
self-bootstrapping from clip0, one as tight as anything in the original grid (channel 34,
MAD 0.07 m), but only 21 of 162 sampled frames pass the match-quality gate, so per-frame
overlap tops out at 2 channels against the 4 a homography needs.

## E. Scope — what may and may not be published

- **May**: clip0 positions in metres, with the C-P3 gate applied and gated rows null.
- **May not**: per-player distance, top speed, or sprint counts for any clip. C-P1 and
  C-P2 both failed, and `02_preregistration_metrics.md` §B binds that outcome to
  withholding those numbers. This is the same rule that kept them unpublished before.
- **clip1 / clip2**: unchanged and unavailable — **P-6 holds**. Neither has a verified
  pitch anchor, and tracking propagates an anchor rather than creating one.
