import numpy as np
import pytest

from footy.calib.line_dlt import homography_from_lines, project, residuals_px
from footy.calib.pitch2023 import CIRCLES, STRAIGHT_LINES

# A plausible broadcast-ish pitch->image homography (main camera near halfway).
H_TRUE = np.array(
    [
        [9.0, 3.2, 60.0],
        [0.4, -2.1, 640.0],
        [0.001, 0.006, 1.0],
    ]
)

NAMES = [
    "Middle line",
    "Side line top",
    "Side line bottom",
    "Big rect. left main",
    "Big rect. left top",
]


def segments_for(names):
    img, pit = [], []
    for n in names:
        a, b = STRAIGHT_LINES[n]
        pa, pb = np.array(a, dtype=float), np.array(b, dtype=float)
        ia, ib = project(H_TRUE, np.stack([pa, pb]))
        img.append((ia, ib))
        pit.append((pa, pb))
    return img, pit


def test_recovers_known_homography_from_lines():
    img, pit = segments_for(NAMES)
    H = homography_from_lines(img, pit)
    probe = np.array([[10.0, 10.0], [52.5, 34.0], [95.0, 60.0]])
    np.testing.assert_allclose(project(H, probe), project(H_TRUE, probe), atol=1e-6)


def test_recovery_is_exact_even_from_clipped_endpoints():
    """Annotated endpoints are frame clips, not landmarks: shifting endpoints along
    the line must not change the solution."""
    img, pit = segments_for(NAMES)
    shifted = []
    for (a, b), _ in zip(img, pit, strict=True):
        d = (b - a) / np.linalg.norm(b - a)
        shifted.append((a + 17.0 * d, b - 5.0 * d))
    H = homography_from_lines(shifted, pit)
    probe = np.array([[30.0, 20.0], [70.0, 50.0]])
    np.testing.assert_allclose(project(H, probe), project(H_TRUE, probe), atol=1e-6)


def test_residuals_near_zero_for_true_h_including_circles():
    elements = {}
    for n in NAMES:
        a, b = STRAIGHT_LINES[n]
        t = np.linspace(0, 1, 5)[:, None]
        seg = np.array([a]) * (1 - t) + np.array([b]) * t
        elements[n] = project(H_TRUE, seg)
    (cx, cy), r = CIRCLES["Circle central"]
    th = np.linspace(0, 2 * np.pi, 11)
    rim = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])
    elements["Circle central"] = project(H_TRUE, rim)

    res = residuals_px(H_TRUE, elements)
    for name, d in res.items():
        assert d.max() < 0.35, f"{name}: {d.max()}"  # dense-polyline discretisation only


def test_too_few_lines_raises():
    img, pit = segments_for(NAMES[:3])
    with pytest.raises(ValueError, match=">=4"):
        homography_from_lines(img, pit)
