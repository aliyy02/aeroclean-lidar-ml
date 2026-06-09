"""Parse a test_bed_multiwindow.py ground-truth .txt into windows + plane + pose.

Units in the file are mm; the parser returns meters (the cloud is in meters).
"""
import numpy as np

from calibration.gt_parse import parse_gt

SNIPPET = """\
6-DOF Test Bed - Multi-Window Measurement
========================================================================

Pose: X=665.0 Y=601.0 Z=376.0 | Yaw=0.0 Pitch=0.0 Roll=0.0
LiDAR (bed frame): [-577.17, -275.30, -713.58]

Window corners in LiDAR frame (mm):
  r1c1:
    UL: [1645.17, -3134.70, -3087.42]
    UR: [1645.17, -1504.70, -3087.42]
    LL: [1645.17, -3134.70, -1933.42]
    LR: [1645.17, -1504.70, -1933.42]
  r1c2:
    UL: [1645.17, -1432.70, -3087.42]
    UR: [1645.17, 197.30, -3087.42]
    LL: [1645.17, -1432.70, -1933.42]
    LR: [1645.17, 197.30, -1933.42]

Facade plane: -1.0000x + 0.0000y + 0.0000z + 1645.1700 = 0
"""


def test_parses_all_windows_and_corners():
    gt = parse_gt(SNIPPET)
    assert set(gt.windows) == {"r1c1", "r1c2"}
    assert set(gt.windows["r1c1"]) == {"UL", "UR", "LL", "LR"}


def test_corners_converted_mm_to_meters():
    gt = parse_gt(SNIPPET)
    np.testing.assert_allclose(
        gt.windows["r1c1"]["UL"], [1.64517, -3.13470, -3.08742], atol=1e-6)
    np.testing.assert_allclose(
        gt.windows["r1c2"]["LR"], [1.64517, 0.19730, -1.93342], atol=1e-6)


def test_plane_x_and_normal():
    gt = parse_gt(SNIPPET)
    assert abs(gt.plane_x - 1.64517) < 1e-6           # facade depth, meters
    np.testing.assert_allclose(gt.plane_normal, [-1.0, 0.0, 0.0], atol=1e-9)


def test_pose_and_lidar_bed_position():
    gt = parse_gt(SNIPPET)
    assert gt.pose["X"] == 665.0 and gt.pose["yaw"] == 0.0
    np.testing.assert_allclose(gt.lidar_bed, [-0.57717, -0.27530, -0.71358], atol=1e-6)


def test_window_grid_bbox_meters():
    gt = parse_gt(SNIPPET)
    (y0, y1), (z0, z1) = gt.grid_bbox()
    assert abs(y0 - (-3.13470)) < 1e-6 and abs(y1 - 0.19730) < 1e-6
    assert abs(z0 - (-3.08742)) < 1e-6 and abs(z1 - (-1.93342)) < 1e-6
