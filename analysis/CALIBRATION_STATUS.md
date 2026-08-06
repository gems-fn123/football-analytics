# Per-clip field registration — status

Goal: a general algorithm that builds the digital field model for *any* camera angle,
so a new stadium does not need hand-tuning.

## What now works

**Line evidence extraction is general and stable across clips.** Three fixes made it so:

1. **Field mask** — take the largest connected grass component and fill holes by contour
   rather than a large morphological close. The close was bridging the touchline into
   the stands, which put the entire crowd inside the "pitch".
2. **Line response** — a morphological white top-hat on the Lab L channel, not absolute
   colour thresholds. Sunlit grass is brighter than a shaded line, so colour thresholds
   cannot work; a top-hat responds to *thin bright ridges* regardless of illumination.
3. **Threshold** — robust and noise-adaptive (`median + 4·MAD` of the in-field ridge
   response), followed by an elongation filter that keeps only line-like components.
   The earlier percentile threshold was the real bug: it returns a fixed *fraction* of
   the pitch as "line" whether or not any line is present, so the mask filled with
   mowing stripes and shadow edges.

Result: comparable, mostly-genuine line pixels across very different clips
(clip0 1 487 · clip1 4 359 · clip2 2 787).

**Two objective bugs found and fixed:**

- **Tilt sign.** Solving the rotation chain for "look-at target lands on the principal
  point" gives `tan(tilt) = −Cz / horiz`, i.e. a camera above the pitch has *negative*
  tilt in this convention. The bounds forced `tilt ≥ 0.02`, so the optimiser was locked
  out of every downward-looking camera. Verified fixed: look-at now projects to the
  principal point exactly for all test configurations.
- **Coverage is gameable.** Ranking by "fraction of detected line pixels explained"
  rewards *shrinking* the pitch — a half-scale pitch packs more markings into frame, so
  every detected pixel sits near some projected line. Replaced with the F1 of precision
  (model markings supported by evidence) and recall (evidence explained by the model).
  Verified: F1 ranks the trusted registration above the half-scale impostor
  (0.339 vs 0.235), where recall alone ranked them the wrong way round (0.707 vs 0.716).

## What does not yet work

**Automatic registration does not reliably converge on sparse views.** On clip0 the
automatic fit reaches F1 0.377 against the verified-correct 0.339 — it finds a pose that
explains the line evidence *better* than the right answer while sitting 490 px away.
More search will not fix this: the objective itself does not uniquely identify the pose
from this evidence.

The missing ingredient is **semantic constraint**. The centre circle is unique on a
pitch; binding the detected ellipse to it pins scale and position immediately. Line
evidence alone is ambiguous because a pitch is largely a set of parallel lines, and many
poses explain a touchline plus a perpendicular line equally well.

## Per-clip geometry available

| Clip | Reference frame | Visible geometry | Calibratable? |
|---|---|---|---|
| clip0 | 0 | halfway line, far touchline, centre circle | **Yes** — seeded registration verified, median 1.9 px, used for the match report |
| clip1 | 0 | halfway line + far touchline **only** | **No.** Two lines is 4 constraints for an 8-DOF homography. The other strong responses are a shadow boundary and the iStock watermark. Needs a different reference frame, later in the clip where the camera pans onto a penalty area |
| clip2 | 182 | far touchline, penalty area front and side lines, goal | **Yes, pending** — landmarks identified below, not yet fitted |

### clip2 landmarks (read against the detected line fits)

The magenta evidence traces a clear V at the near corner of the right-hand penalty area:

| Pitch (m) | Image (px) | Feature |
|---|---|---|
| (88.5, 13.84) | (439, 377) | penalty area near corner — the V vertex |
| (88.5, 68) | (157, 177) | penalty front line ∧ far touchline, both extended |
| (105, 13.84) | (715, 291) | penalty side line meeting the goal line |

Three labelled lines are available — `x = 88.5`, `y = 13.84`, `y = 68` — giving 6 of the
8 constraints. A fourth independent correspondence, or a line-residual fit rather than a
4-point solve, closes it.

## Next steps, in order of leverage

1. **Bind the detected ellipse to the centre circle.** This is the single change most
   likely to make automatic registration work, because it is the one unambiguous
   landmark on the pitch. The current ellipse fitter fails and needs rewriting.
2. **Fit from labelled line correspondences** (point-to-line residuals) rather than a
   4-point solve, so 3 lines plus chamfer suffices — enough for clip2 today.
3. **Choose the reference frame automatically** by scoring frames on how much distinct
   geometry they contain, instead of assuming frame 0. This alone fixes clip1.
