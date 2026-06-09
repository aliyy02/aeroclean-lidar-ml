"""estimate_roll recovers the boresight roll that levels the panel to the GT grid."""
import numpy as np

from calibration.gt_parse import GroundTruth
from calibration.frames import apply_roll
from calibration.roll_estimate import estimate_roll

X0 = 1.645


def make_gt(y=(-1.5, 1.5), z=(-1.0, 0.0)):
    win = {"UL": np.array([X0, y[0], z[0]]), "UR": np.array([X0, y[1], z[0]]),
           "LL": np.array([X0, y[0], z[1]]), "LR": np.array([X0, y[1], z[1]])}
    return GroundTruth(windows={"r1c1": win}, plane_normal=np.array([-1.0, 0, 0]),
                       plane_d=X0, pose={}, lidar_bed=np.zeros(3))


def filled_panel(gt, npts=60):
    (y0, y1), (z0, z1) = gt.grid_bbox()
    yy, zz = np.meshgrid(np.linspace(y0, y1, npts), np.linspace(z0, z1, npts))
    yy, zz = yy.ravel(), zz.ravel()
    return np.column_stack([np.full(yy.shape, X0), yy, zz])


def test_recovers_correction_for_known_roll():
    gt = make_gt()
    raw = apply_roll(filled_panel(gt), 15.0)        # simulate a +15 deg sensor roll
    alpha = estimate_roll(raw, gt)
    assert abs(alpha - (-15.0)) < 1.0               # correction un-rolls it


def test_zero_for_already_level_panel():
    gt = make_gt()
    alpha = estimate_roll(filled_panel(gt), gt)
    assert abs(alpha) < 1.0


def test_recovers_negative_roll():
    gt = make_gt()
    raw = apply_roll(filled_panel(gt), -20.0)
    alpha = estimate_roll(raw, gt)
    assert abs(alpha - 20.0) < 1.0
