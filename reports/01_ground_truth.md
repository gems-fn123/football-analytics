# Ground truth — manual person counts

Collected **before** the detector was run on full clips, per the protocol in
`00_preregistration.md` §5. Counted by eye from 2.4× upscaled stills, split into left
and right halves so individual figures are resolvable.

## Counting rule

Counted: players, goalkeepers, and match officials whose feet are **inside the field
of play** (touchlines inclusive).
Excluded: spectators, stewards, photographers, anyone behind the advertising boards.
That boundary is the whole point — it is exactly what criterion **C-B3** tests, since
a COCO `person` detector does not know the boundary exists.

## Counts

| Clip | Frame | Home-kit | Away-kit | GK | Officials | **Total on pitch** | Uncertainty |
|---|---|---|---|---|---|---|---|
| clip0 | 0 | 6 blue | 8 white | 0 visible | 2 (yellow) | **15** | ±2 |
| clip1 | 440 | 9 red | 10 blue | 0 clearly visible | 1 (olive) | **20** | ±3 |
| clip2 | 182 | 7 red | 9 cyan | 1 (green, in goal) | 1 (yellow) | **17** | ±3 |

Uncertainty is genuine, not decorative: at 768×432 a distant player is ~20 px tall,
figures overlap near the far touchline, and the watermark obscures frame centre. Two
of the frames have players bisected by the left/right split, resolved by comparing
absolute x positions across halves.

## Immediate observations

1. **The counts bracket the repo's own expectation.** `configs/camera/broadcast.yaml`
   declares `players_visible: 15`. Measured: 15, 20, 17. The pre-registered recall
   floor of 10 (C-B1) is therefore a real bar — reachable, but not free.

2. **The crowd hazard is visually severe.** All three frames show densely packed
   stands containing *thousands* of human figures — far more people off the pitch than
   on it. If the detector had no effective size/confidence filter, person counts would
   run into the hundreds rather than the teens. This is what C-B3 exists to catch.

3. **Shirt numbers are not legible to a human at this resolution**, let alone to OCR.
   A "9" is faintly discernible on one white shirt in clip0 at 2.4× upscale and
   nowhere else. Identity resolution is expected to abstain completely — and abstention
   is the correct behaviour, not a failure.

4. **Officials are colour-distinct** (yellow in clip0/clip2, olive in clip1), which is
   what the team stage's `referee_colour` matching depends on.
