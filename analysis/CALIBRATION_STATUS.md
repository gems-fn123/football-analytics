# Per-clip field registration — status

Goal: a general algorithm that builds the digital field model for *any* camera angle,
so a new stadium does not need hand-tuning.

Everything below is measured against clip0, whose hand-seeded registration is verified to
a median of 1.9 px and is used as ground truth. Numbers quoted from a single run are
marked as such, because the search turned out to have far more variance than single runs
suggest — see "How reliable is this really".

## What now works

**Line evidence extraction is general and stable across clips.** Four fixes made it so:

1. **Field mask** — take the largest connected grass component and fill holes by contour
   rather than a large morphological close. The close bridged the touchline into the
   stands, putting the entire crowd inside the "pitch".
2. **Line response** — a morphological white top-hat on the Lab L channel, not absolute
   colour thresholds. Sunlit grass is brighter than a shaded line, so colour thresholds
   cannot work; a top-hat responds to *thin bright ridges* regardless of illumination.
3. **Hysteresis, not a single threshold.** Measured along clip0's known centre circle,
   the ridge response runs from 3 to 67 — one marking is not one brightness. Any level
   that excludes grass texture also chops the circle into arcs too short to survive a
   length filter. Seeding at `median + 4·MAD` and growing through connected pixels above
   `median + 2·MAD` recovers whole markings without starting anywhere weak.
4. **Shape filter measures thinness, not straightness.** The previous filter demanded 3:1
   PCA elongation, which is a prior on *being straight*. A closed rim is as wide as it is
   tall, so the centre circle scored ~1:1 and was deleted every time. Measuring the
   response along the known circle showed 57% of it clearing the intensity threshold and
   0% reaching the mask, which located the loss precisely. Now: stroke width from the
   distance transform (95th percentile, so one fat crossing does not delete a whole
   line), and centreline length as area over stroke.

Measured effect on clip0: centre-circle recall **0% → 58%**, halfway line 83%.

**Conic fitting is algebraic and verified.** `cv2.fitEllipse` fits the *contour* of a
blob — a painted line is a ridge a few pixels thick, so its contour runs up one side of
the arc and back down the other — and it cannot fit a partial arc at all. Replaced with
the Halir–Flusser direct ellipse fit, checked in `test_conics.py` against synthetic
ellipses (exact recovery), a 110° arc, and collinear points.

That test records an honest limitation: **a short arc's extrapolated size is not
recoverable.** The algebraic fit is biased toward low eccentricity and returns a
semi-major axis ~25% short on a 110° arc. It still *hugs* the observed arc to 0.8 px, so
it selects the right pixels — which is all the pipeline asks of it. Nothing downstream
uses the fitted conic's geometry.

**Three ways of accidentally deleting the circle, all found by measurement:**

- **The elongation filter** (above) — 100% loss.
- **Line erasure** — the conic search blanked out detected lines first, so that only
  curved evidence remained. But the bottom of a projected centre circle is nearly
  straight over a long span, so Hough detects it as a line: erasure destroyed **97%** of
  clip0's circle (1034 px → 30 px). Replaced with a **bow test** — the peak sideways
  deviation of the evidence under a fitted line — which separates a painted line
  (bow ≤ 1.5 px) from a chord of an arc (bow 2.6–5.2 px) with no prior knowledge of what
  markings are present. Straight lines are now *discounted at scoring time* instead of
  erased, so the circle survives while its pixels still are not double-counted.
- **Static-overlay removal** — see below; it is off by default for the same reason.

**Circle candidates, not a circle.** No local score reliably picks the centre circle out
of a line mask, and this is not a tuning failure. Only 58% of clip0's rim clears
detection, so a conic sweeping across several markings genuinely explains more evidence
than the true circle does — scored 16.7 against the true rim's 6.2. The detector
therefore returns the top 8 distinct candidates. On clip0 the true rim is **candidate #6
of 8** (81% of its pixels genuinely on the circle, and the only candidate whose axis
ratio, 3.78, is near the true 4.4).

**Three fixes that each mattered a lot, on a single-run basis:**

- **One-directional circle residual.** The symmetric chamfer also asks every model-circle
  point to find a rim pixel, and a real detection covers half the rim, so that term is
  minimised by shrinking the model circle onto the observed arc.
- **Tight truncation on the circle term** (8 px, against 22 px for lines). The best real
  candidate is 81% pure, and at the line term's truncation that 19% of contamination
  dragged the pose from 56 px to 426 px of error.
- **A camera plausibility prior** — height 8–40 m, and the pitch larger than the frame.
  Labelled as a prior in the code. The second half matters because F1 does not catch a
  shrunken pitch: markings end up scattered everywhere, so most detected pixels land near
  *some* projected line, and a pose with the whole pitch crammed into a fifth of the
  frame scored F1 0.442 against the correct pose's 0.372.

## How reliable is this really

**Not yet reliable, and the single-run numbers above oversell it.** Repeating the fit over
six seeds on clip0:

| configuration | per-seed error vs verified reference (px) | median | picked by best quality |
|---|---|---|---|
| lines only | 517 · 550 · 638 · 461 · 277 · 762 | 533 | 277 |
| all 8 circle candidates | 415 · 625 · 598 · 447 · 240 · 304 | 431 | 447 |
| the *true* rim, handed to it | 42 · 47 · 458 · 512 · 449 · **16** | 248 | 512 |
| lines + player heights | 738 · 429 · 638 · 711 · 577 · 651 | 644 | 638 |
| candidates + player heights | 869 · 419 · 758 · 431 · 49 · 621 | 526 | 758 |
| true rim + player heights | 41 · 464 · 460 · 138 · 180 · **16** | 159 | **16** |

Three things follow, and the second is the important one.

1. The circle constraint genuinely works **when the search lands in the right basin** —
   16 to 47 px, which is a usable registration. Given the true rim it does so about half
   the time.
2. **The objective does not identify the correct pose.** Its self-reported quality is
   *anti*-correlated with true error: over the third row the 16 px fit scored 0.31 and
   the 512 px fit scored 0.35. So picking the best-scoring of several seeds selects the
   *worst* answer, and no amount of extra search fixes it.
3. **Player-height evidence repairs the ranking, but only where the geometry is already
   right.** Added to the true-rim case it cuts the median from 248 px to 159 px and — the
   point of adding it — makes best-quality selection pick the 16 px answer instead of the
   512 px one, which is the failure in (2) fixed. Added to *detected* rims it makes things
   worse (lines 533 → 644, candidates 431 → 526).

That is a negative result about the objective, not about the search, and it supersedes
the earlier note that "more search will not fix this" — now confirmed with the circle
constraint included and quantified over repeats. The remaining gap is *detection*, not
scoring: with the right rim there is now a term that recognises the right answer.

### Why the player term helps only there: it constrains scale, not position

The reason it splits that way was a guess until `probe_players.py` scored the verified
pose against deliberately broken ones:

| pose | score | implied player height |
|---|---|---|
| verified | **0.910** | 1.98 m |
| half the focal length | 0.011 | 3.50 m |
| double the focal length | 0.082 | 1.09 m |
| camera 20 m higher | 0.000 | 6.00 m |
| camera at half the height | 0.028 | 0.99 m |
| **camera slid 30 m along the pitch** | **0.910** | 1.98 m |

Every error of *scale* is crushed — two orders of magnitude. The last row scores
**exactly** as well as the truth. Sliding the camera along the pitch is a null direction
of this term: the feet are mapped to the ground through the same wrong homography, so the
player is placed wherever that homography says and the height relationship stays
self-consistent. The `on_pitch` guard does not catch it either, since 30 m along a 105 m
pitch still leaves everyone inside the touchlines.

This corrects a claim made here earlier — that the term catches "a shrunken pitch *and* a
slid pitch". It catches the first and is blind to the second. And the blind direction is
precisely the one a set of near-parallel lines is already ambiguous in, which is why the
term rescues the oracle case (the circle has pinned position, leaving only scale in doubt)
and does nothing for detected rims (position is wrong too, and nothing here can tell).

The practical reading: **position must come from geometry; the player term can only be a
scale check on top of it.** That is what moves cone–conic decomposition to the top of the
list below — it fixes position outright rather than asking the search to find it.

## The accuracy floor is 1.1 px — so these are real errors

Worth checking before trusting any of the numbers above. The verified reference was solved
as an unconstrained 8-DOF homography from five hand-labelled points, so nothing forced it
to be a physically realisable camera — and strictly it is not one: decomposing it gives an
imaginary focal length (f² = −580 022), so `params_from_homography` refuses it. The pose
search only ever proposes real cameras, so in principle every figure above could have been
dominated by that mismatch rather than by search error.

Measured with `fit_params_to_homography`, which fits the closest 7-DOF camera by
reprojection: the best reachable camera sits **1.1 px** from the reference over the visible
pitch. So the mismatch is negligible and the errors above are genuine. The fit also names
the pose the search is actually hunting for — camera 13.8 m outside the touchline near the
halfway line, 9.9 m up, focal 954 px, tilted 8° down — which is a more diagnostic target
than a pixel count, since it says *which parameter* a failed run got wrong.

The decomposition itself is verified separately: `homography_from` → `params_from_homography`
→ `homography_from` round-trips to 5×10⁻¹⁶.

**What it means practically:** clip0's analytics continue to run off the verified
hand-seeded registration in `calib/clip0_reference.json`. Automatic registration is not
yet trustworthy enough to replace it, and the honest reason is written down rather than
hidden behind a single good-looking run.

## Static overlay removal — measured, and off by default

A burnt-in graphic (the iStock watermark here, a score bug in real footage) is fixed to
the *image*, while pitch markings are fixed to the *ground*, so persistence across a
panning shot should separate them. Implemented in `static_overlay_mask`, and it does
remove ~3 000 px of watermark.

It is off by default because the measured cost exceeds the benefit: centre-circle recall
drops 58% → 42% and touchline recall 17% → 3%. The reason is a real limitation rather
than a tuning problem — clip0 pans *horizontally*, and a structure parallel to the pan
(the touchline above all) stays at the same image height throughout, so it looks exactly
as persistent as a graphic. Testing "fixed to the ground" properly means warping frames
into a common ground frame first, not testing "fixed to the image" and hoping the camera
moved the right way.

One implementation note worth keeping: the removal must be applied to the *final* mask.
Removing overlay pixels before hysteresis cuts long markings into fragments that the
length filter then deletes, turning a 20% pixel loss into a 65% loss of evidence.

## Per-clip geometry available

| Clip | Reference frame | Visible geometry | Calibratable? |
|---|---|---|---|
| clip0 | 0 | halfway line, far touchline, centre circle | **Yes, by hand.** Seeded registration verified at 1.9 px median and used for the match report. Automatic: unreliable, see above |
| clip1 | 0 | halfway line + far touchline **only** | **No.** Two lines is 4 constraints for an 8-DOF homography. The other strong responses are a shadow boundary and the watermark. Needs a different reference frame, later in the clip, where the camera pans onto a penalty area |
| clip2 | 182 | far touchline, penalty area front and side lines, goal | **Yes, pending** — landmarks identified below, not yet fitted |

### clip2 landmarks (read against the detected line fits)

The evidence traces a clear V at the near corner of the right-hand penalty area:

| Pitch (m) | Image (px) | Feature |
|---|---|---|
| (88.5, 13.84) | (439, 377) | penalty area near corner — the V vertex |
| (88.5, 68) | (157, 177) | penalty front line ∧ far touchline, both extended |
| (105, 13.84) | (715, 291) | penalty side line meeting the goal line |

Three labelled lines are available — `x = 88.5`, `y = 13.84`, `y = 68` — giving 6 of the
8 constraints. A fourth independent correspondence, or a line-residual fit rather than a
4-point solve, closes it.

## Next steps, in order of leverage

1. **Pose from a circle in closed form.** Now the highest-leverage item, because the
   measurements above localise the fault to *which rim gets detected*, and this removes
   the need to guess. A projected circle known to be 9.15 m on the ground plane determines
   the camera up to a small discrete ambiguity (cone–conic decomposition). That replaces
   random sampling entirely: each of the 8 candidates yields a handful of exact poses to
   score, instead of a basin the search has to find. It also turns the player-height term
   from a tie-breaker that needs luck into the discriminator over a short candidate list —
   which is the one configuration measured to rank correctly.
2. **Fit from labelled line correspondences** (point-to-line residuals) rather than a
   4-point solve, so 3 lines plus chamfer suffices — enough for clip2 today.
3. **Choose the reference frame automatically** by scoring frames on how much distinct
   geometry they contain, instead of assuming frame 0. This alone fixes clip1.

Done and superseded: *add evidence the line objective does not have* — implemented as
`player_score`, measured over seeds, results in the table above. It works where the
geometry is already close and not otherwise, which is why (1) moved to the top.

## Probes

Each of these is a measurement, not a demo, and each exists because a guess turned out to
be wrong:

| script | question it answers |
|---|---|
| `test_conics.py` | does the conic maths do what it claims, on cases with known answers |
| `probe_response.py` | how much of a *known* marking reaches the mask |
| `probe_overlay.py` | does overlay removal remove more clutter than evidence |
| `probe_erase.py` | how much of the circle does line erasure destroy |
| `probe_binding.py` | is the detector or the constraint at fault |
| `probe_rank.py` | which measurable quantity tracks true pose error |
| `probe_stability.py` | is a good result reproducible or a lucky seed |
| `probe_ellipse.py` | is the true rim among the candidates the detector returns |
| `probe_players.py` | does the player-height term score the verified pose above wrong ones |
| `probe_floor.py` | how much of the measured error is model mismatch rather than search |
