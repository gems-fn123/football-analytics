# Results — metric chain

Scored against `02_preregistration_metrics.md`, which was frozen before any of this ran.
Criteria that failed are reported as failed.

## Headline

**Calibration is solved. It is no longer the thing blocking metres.**
**Tracking was the next binding constraint and has been substantially improved.**
**Neither was sufficient: the physical criteria still fail, and the numbers stay unpublished.**

The first reversal is the main finding. Before this work the assumption — mine included,
and the reason tasks 17 and 20 existed — was that registration accuracy was the obstacle.
It is not.

The second is the more useful one for whoever picks this up: having fixed calibration,
then the velocity estimator, then identity switches, then the association space, the
median player speed did not move at all (3.75–3.77 m/s throughout). Four independent
interventions on the mechanisms that *should* have caused it changed nothing, which is
strong evidence the remaining gap is neither geometry nor tracking, but the combination
of 38 px players and a 32 s all-action clip that full-match baselines do not describe.

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

## C-bis. Tracking, attacked directly

Two approaches, both scored. Criteria C-S1 / C-S2 were frozen in advance.

### Appearance re-ID cannot work at this resolution — measured, not assumed

Before believing any merge, the embedding was validated without needing labels: crops
from the **same** segment are the same player by construction, and crops from segments
that **overlap in time** are provably different people, since one person cannot be in
two places in one frame.

| | cosine similarity |
|---|---|
| same player | median 0.888 |
| provably different player | median 0.804 |
| best achievable balanced accuracy | **77.6 %** (at threshold 0.86) |

49.4 % of provably-different pairs score above the 10th percentile of same-player pairs.
Player boxes are a median of **38 px** tall against OSNet's 256×128 input, so the crops
are heavily upscaled and carry little identity. OSNet weights loaded correctly (99.6 % of
parameters matched), so this is the footage, not the setup.

| criterion | bar | result | |
|---|---|---|---|
| C-S2 no same-frame duplicates | 0 | **0** | PASS |
| C-S1 ids ≤ 5× median persons | ≤ 75 | **235** | **FAIL** |

The stitcher's ambiguity test behaved correctly — it made only 16 merges and produced
**zero** provable false merges — but that conservatism is why it cannot fix
fragmentation. It also made the physical criteria slightly *worse* (C-P1 1.68 % → 2.30 %),
because a merge spans a temporal gap and velocity across that gap is not measurable.
**Stitching stays off**, which is what C-S2's framing intended.

### Associating in pitch metres instead of image pixels

ByteTrack associates by IoU in the image while clip0's camera pans 1934 px, so a
stationary player has a large image-space velocity and IoU between consecutive frames
collapses. With the homography validated, association can happen in metres, where the
gate follows from physics rather than tuning: a player covers at most
`12 m/s · dt + 3σ√2` ≈ **1.37 m** between frames, against a 105 m pitch.

| | distinct ids | segments after physics split | median track life |
|---|---|---|---|
| ByteTrack (image space) | 66 | **251** | 1.6 s |
| pitch-space association | **122** | **127** | **2.3 s** |

The second column is the informative one. ByteTrack's 66 ids shatter into 251 physically
honest segments, i.e. most ids contain an identity switch. Pitch-space association yields
122 tracks that survive the same test almost intact (127), so they are internally
consistent rather than merely fewer.

One bug found and fixed here: the association gate tested distance from the *predicted*
position only, so a bad velocity estimate could admit a jump the player could never
physically make. Enforcing the bound from the last *observed* position as well took the
physics split from 122 → 172 down to 122 → 127.

### It did not rescue the physical criteria

| | C-P1 (bar ≤ 1 %) | C-P2 (bar 9–12) |
|---|---|---|
| ByteTrack + physics split | 1.77 % | 24.6 |
| pitch-space association | **2.09 %** | 23.4 |

**Median speed is 3.75–3.77 m/s under every tracking variant tried.** It does not move
when the tracker changes, so the median is not an identity-switch artifact — which rules
out the mechanism P-5 named and the one this section set out to fix.

Nor is it a noise floor: the estimator resolves speeds down to 0.30 m/s (p1), 2.0 % of
frames sit under 0.5 m/s, and 38 % of tracks dip below 1 m/s. A systematic floor would
show none of that.

What remains is a distribution that is simply **hotter than full-match football**: 20 % of
time under 2 m/s where a real match spends 70–85 %. Part of that is genuine — clip0 is
32 s of continuous active play from a kickoff, with none of the stoppages, walking or
standing goalkeepers that full-match baselines average over, so C-P2's 9–12 km/90 bar
does not transfer. Part is not: p99 of 12.4 m/s exceeds the human record, so the tail is
still contaminated by position outliers at 38 px player size.

Both parts are reported rather than resolved. Tuning thresholds until the criteria went
green would have made the frozen criteria worthless.

## C-ter. Is the metre scale grounded, or just assumed?

Worth asking directly, because the homography is solved *to* a pitch model: if the real
pitch differs from the model, every distance and speed scales with the error.

**What the repo assumes**: 105 × 68 m (`configs/pitch/standard_105x68.yaml`). IFAB Law 1
allows 100–110 × 64–75 m for international matches and 90–120 × 45–90 m otherwise;
105 × 68 is the FIFA/UEFA recommendation and the convention StatsBomb and socceraction
normalise to, so it is the right default.

**What is actually grounded.** Scale does not rest on those numbers. Four of the five
seed correspondences in `seed_clip0.py` are the **centre circle**, whose 9.15 m radius is
fixed by the Laws on every pitch in the world and does not scale with pitch size. Only
one correspondence (halfway line meeting the far touchline) uses the assumed width.

Three *independent* invariants can be checked against the verified reference:

| invariant | true value | implied | error |
|---|---|---|---|
| centre circle radius | 9.15 m | 9.77 m | **+6.8 %** |
| pitch width (touchline) | 68 m assumed | 66.0 m | **−2.9 %** |
| player stature | ~1.80 m | 1.98 m | **+10 %** |

They agree to within about 10 %, and **that spread is the honest error bar on the metre
scale** — better than the ±2 m positional bound the project already works to, and far too
small to explain the physical-criteria failures. Correcting for any of them moves median
speed from 3.77 m/s to 3.43–3.88 m/s, against the ~2 m/s a real match implies. **Pitch
dimensions are not the cause.**

**Length is the least constrained, and least important.** Clip0 shows no goal line, so
105 m is unconstrained by this footage. It matters less than it looks: length enters as
an origin offset, not as a metric scale, so distances and speeds are unaffected. It would
matter for zone-based or xT analysis, which is not published here.

**The error is strongly anisotropic, and this is the useful part.** Reprojecting the
detected circle through the verified reference:

| circle point | implied radius | error |
|---|---|---|
| left | 9.02 m | −1.4 % |
| right | 9.02 m | −1.4 % |
| near | 8.56 m | −6.5 % |
| **far** | **12.49 m** | **+36.5 %** |

Lateral geometry is excellent; **depth is poor and gets worse with distance**. That shows
up downstream exactly as it should: |v_y| / |v_x| = 1.18 overall, and |v_y| climbs from
3.24 m/s in the nearest depth band to 4.30 m/s in the furthest while |v_x| stays flat.
Real players have no reason to move faster across the pitch than along it.

**One artefact worth recording so nobody repeats it.** Fitting a homography from the four
circle points *alone* — which is exactly determined, 4 points giving 8 equations — puts
the far touchline at 16.5 m from centre, implying a 33 m pitch. That is not a measurement,
it is ill-conditioning: extrapolating depth from a circle spanning only 81 px vertically
has no redundancy to absorb the ~9 px error on the far point, which is itself 36 % off.
The five-point fit, which includes the touchline, gives the sane 66 m. Depth extrapolation
from a small baseline is the weak direction of this whole setup.

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
