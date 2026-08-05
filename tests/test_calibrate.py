import numpy as np

from footy.stages.calibrate import Calibrator


def test_apply_identity_homography():
    H = np.eye(3)
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_allclose(Calibrator.apply(H, pts), pts)


def test_apply_translation():
    H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    pts = np.array([[0.0, 0.0], [2.0, 2.0]])
    np.testing.assert_allclose(Calibrator.apply(H, pts), [[5.0, -3.0], [7.0, -1.0]])
