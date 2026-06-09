"""Offline test for the segmentation/measurement core (no ROS, no sim).

Builds a synthetic frontal facade with two windows + a mullion gap, places known
glass/frame returns, inverts the native->corrected transform to fake a native
cloud, and checks segment() recovers class, incidence angle, and depth-behind.
"""
import numpy as np

import sys
sys.path.insert(0, ".")
from calibration.gt_parse import parse_gt           # noqa: E402
from calibration.io_bag import Frame                 # noqa: E402
from analysis.seg import segment, GLASS, FRAME       # noqa: E402

GT = """6-DOF Test Bed
Pose: X=0.0 Y=0.0 Z=0.0 | Yaw=0.0 Pitch=0.0 Roll=0.0
LiDAR (bed frame): [0.0, 0.0, 0.0]
  r1c1:
    UL: [2000.0, -600.0, -500.0]
    UR: [2000.0, -100.0, -500.0]
    LL: [2000.0, -600.0, 500.0]
    LR: [2000.0, -100.0, 500.0]
  r1c2:
    UL: [2000.0, 100.0, -500.0]
    UR: [2000.0, 600.0, -500.0]
    LL: [2000.0, 100.0, 500.0]
    LR: [2000.0, 600.0, 500.0]
Facade plane: -1.0000x + 0.0000y + 0.0000z + 2000.0000 = 0
"""


def ned_to_native(ned):
    # axis_map: ned_x=nz, ned_y=nx, ned_z=ny  ->  native=(ned_y, ned_z, ned_x)
    return np.column_stack([ned[:, 1], ned[:, 2], ned[:, 0]])


def test_segment_classifies_and_measures():
    gt = parse_gt(GT)
    # corrected-NED returns we want to plant:
    glass = [2.2, -0.35, 0.0]    # in window1, 0.2 m behind plane, ~8.9 deg incidence
    frame = [2.0, 0.0, 0.0]      # in the mullion gap, on the plane, normal incidence
    ned = np.array([glass, frame] * 30, dtype=float)   # duplicate for stable fit/medians
    native = ned_to_native(ned)
    frames = [Frame(stamp_ns=0, xyz=native, intensity=np.full(len(native), 100.0))]

    s = segment(frames, gt, alpha=0.0, ysign=1)
    # row 0 = glass, row 1 = frame
    assert s.cls[0] == GLASS and s.cls[1] == FRAME
    # depth behind GT plane
    assert abs(s.behind_gt[0] - 0.2) < 1e-6
    assert abs(s.behind_gt[1] - 0.0) < 1e-6
    # incidence angle vs GT normal
    assert abs(np.degrees(s.theta_gt[1]) - 0.0) < 1e-6          # straight ahead
    assert abs(np.degrees(s.theta_gt[0]) - 8.9) < 0.5           # arccos(2.2/|.|)
    # range
    assert abs(s.R[1] - 2.0) < 1e-6


def test_roll_invariance_of_physics():
    """Incidence angle, range, depth-behind are invariant to the boresight roll."""
    gt = parse_gt(GT)
    ned = np.array([[2.2, -0.35, 0.12], [2.1, 0.3, -0.2]] * 20, dtype=float)
    native = ned_to_native(ned)
    frames = [Frame(stamp_ns=0, xyz=native, intensity=np.full(len(native), 100.0))]
    s0 = segment(frames, gt, alpha=0.0, ysign=1)
    s30 = segment(frames, gt, alpha=30.0, ysign=1)
    assert np.allclose(s0.R, s30.R)
    assert np.allclose(s0.theta_gt, s30.theta_gt, atol=1e-9)
    assert np.allclose(s0.behind_gt, s30.behind_gt, atol=1e-9)
