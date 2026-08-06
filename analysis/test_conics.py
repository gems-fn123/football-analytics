"""Check the conic maths against cases whose answer is known in advance.

Fitting code that is subtly wrong still returns plausible numbers, so the fitter is
checked against synthetic ellipses (exact answer known), against a partial arc (the case
`cv2.fitEllipse` cannot do at all), and against collinear points (which must be
*rejected*, not fitted).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conics  # noqa: E402


def make(cx, cy, a, b, ang, t0=0.0, t1=2 * np.pi, n=120, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(t0, t1, n)
    ca, sa = np.cos(ang), np.sin(ang)
    u, v = a * np.cos(t), b * np.sin(t)
    p = np.column_stack([cx + u * ca - v * sa, cy + u * sa + v * ca])
    return p + rng.normal(0, noise, p.shape)


def check(name, ok, detail="") -> bool:
    d = f"  {detail}" if detail != "" else ""
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{d}")
    return bool(ok)


SHAPE = (540, 960)


def main() -> int:
    good = True

    print("full ellipse, noise-free: exact recovery")
    truth = make(400, 250, 180, 40, np.deg2rad(12))
    C = conics.fit_conic(truth)
    (cx, cy), (a, b), ang = conics.conic_geometry(C)
    good &= check("centre", abs(cx - 400) < 0.5 and abs(cy - 250) < 0.5, f"({cx:.2f},{cy:.2f})")
    good &= check("axes", abs(a - 180) < 0.5 and abs(b - 40) < 0.5, f"({a:.2f},{b:.2f})")
    # Axis direction is defined mod 180 deg, so 12 and 192 are the same ellipse.
    good &= check("angle (mod 180)", abs(np.rad2deg(ang) % 180 - 12) < 0.5, f"{np.rad2deg(ang):.2f} deg")
    good &= check("support of a full rim", conics.angular_support(C, truth) > 0.95)

    print("partial arc, 1 px noise: the case cv2.fitEllipse cannot take at all")
    p = make(400, 250, 180, 40, np.deg2rad(12), 0.4, 0.4 + np.deg2rad(110), n=90, noise=1.0)
    C = conics.fit_conic(p)
    good &= check("an ellipse is returned", C is not None and conics.conic_geometry(C) is not None)
    # The *extrapolated* geometry of a short arc is not recoverable and is not relied on:
    # the algebraic fit is known to be biased toward lower eccentricity. What downstream
    # code needs is that the fitted curve hugs the observed arc, so that it selects the
    # right pixels - that is what is asserted here.
    d = conics.sampson(C, p)
    good &= check("curve hugs the observed arc", float(np.median(d)) < 1.5, f"median {np.median(d):.2f} px")
    sup = conics.angular_support(C, p)
    good &= check("support reports an arc, not a full rim", 0.15 < sup < 0.60, f"{sup:.2f}")

    print("collinear points must be rejected")
    line = np.column_stack([np.linspace(50, 700, 200), np.linspace(80, 300, 200)])
    line += np.random.default_rng(1).normal(0, 0.7, line.shape)
    C = conics.fit_conic(line)
    good &= check("validate() rejects it", C is None or conics.validate(C, line, SHAPE) is None)
    if C is not None and conics.conic_geometry(C) is not None:
        # Recording *why* it is rejected: angular support does not do it. A line is the
        # rim of an arbitrarily eccentric ellipse, so support comes out high; the cloud
        # being one-dimensional is the discriminating fact.
        print(f"        support {conics.angular_support(C, line):.2f} "
              f"(high - not the discriminator), straightness "
              f"{conics.straightness(line):.4f} (low - this is)")

    print("a genuine rim passes the same gate")
    rim = make(400, 250, 180, 40, np.deg2rad(12), n=200, noise=0.8, seed=3)
    C = conics.fit_conic(rim)
    m = conics.validate(C, rim, SHAPE)
    good &= check("validate() accepts it", m is not None)
    if m:
        good &= check("straightness well above the line case", m["straightness"] > 0.15,
                      f"{m['straightness']:.3f}")

    print("sampson distance is a true distance near the curve")
    C = conics.fit_conic(make(300, 300, 100, 100, 0.0))  # a circle: answer known exactly
    probes = np.array([[300.0, 300.0 + 100 + t] for t in (0.0, 1.0, 3.0, 7.0)])
    d = conics.sampson(C, probes)
    good &= check("distances recovered", bool(np.allclose(d, [0, 1, 3, 7], atol=0.25)),
                  str(np.round(d, 3).tolist()))

    print("sample_ellipse round-trips")
    C = conics.fit_conic(make(400, 250, 180, 40, np.deg2rad(12)))
    s = conics.sample_ellipse(C, 64)
    good &= check("samples lie on the conic", float(conics.sampson(C, s).max()) < 1e-6)

    print("\n" + ("ALL PASS" if good else "FAILURES PRESENT"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
