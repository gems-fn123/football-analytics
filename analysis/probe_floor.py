"""How much of the reported registration error is model mismatch rather than search error?

Every accuracy figure in CALIBRATION_STATUS.md is measured against clip0's verified
reference. That reference was solved as an unconstrained 8-DOF homography from 5 hand
labelled points, so it is not obliged to be a *camera* - and it is not one:
`params_from_homography` returns None for it because the focal length comes out
imaginary.

The pose search only ever proposes real cameras. So there is a floor: the best possible
7-DOF camera still disagrees with the reference by some amount, and no search can do
better. Until that floor is known, a "16 px error" cannot be read as either a near-perfect
fit or a mediocre one.

Reported with the same metric used everywhere else (`compare_to`), so the numbers are
directly comparable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import (  # noqa: E402
    fit_params_to_homography,
    homography_from,
    params_from_homography,
)
from run_auto_calib import compare_to  # noqa: E402

SHAPE = (432, 768)


def main() -> int:
    h, w = SHAPE
    cx, cy = w / 2.0, h / 2.0

    ref = json.loads((HERE / "calib/clip0_reference.json").read_text())
    H_ref = np.array(ref["H_pitch_to_image"])

    print("the verified reference, as a camera")
    exact = params_from_homography(H_ref, cx, cy)
    print(f"  exact decomposition: {'a camera' if exact is not None else 'NOT a camera'}")

    print("\nclosest reachable camera (fitting the 7-DOF model to the reference)")
    p = fit_params_to_homography(H_ref, cx, cy, SHAPE)
    if p is None:
        print("  FAIL: no pose found")
        return 1
    names = ["Cx", "Cy", "Cz", "pan", "tilt", "roll", "f"]
    print("  " + "  ".join(f"{n}={v:.2f}" for n, v in zip(names, p)))

    H_fit = homography_from(p, cx, cy)
    floor = compare_to(H_fit, H_ref, SHAPE)
    print(f"\n  FLOOR: {floor:.1f} px median over the visible pitch")

    # A round trip through the decomposition confirms the fitted pose is self-consistent,
    # i.e. the floor is the model's limit and not an artefact of this fit.
    back = params_from_homography(H_fit, cx, cy)
    if back is not None:
        rel = float(np.linalg.norm(np.array(back) - p) / np.linalg.norm(p))
        print(f"  (fitted pose decomposes back to itself, relative error {rel:.2e})")

    print("\ninterpretation, against the measured results")
    for label, err in (("best seed, true rim", 16.0),
                       ("median, true rim + players", 159.0),
                       ("median, all candidates", 431.0),
                       ("median, lines only", 533.0)):
        excess = err - floor
        print(f"  {label:<28} {err:6.0f} px  =  {floor:.0f} floor + {excess:.0f} search")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
