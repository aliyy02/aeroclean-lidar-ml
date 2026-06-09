"""detect_floor finds the dominant horizontal plane below the panel."""
import numpy as np

from calibration.gt_parse import GroundTruth
from calibration.label import detect_floor

X0 = 1.645


def make_gt(y=(-0.5, 0.5), z=(-1.0, 0.0)):
    win = {"UL": np.array([X0, y[0], z[0]]), "UR": np.array([X0, y[1], z[0]]),
           "LL": np.array([X0, y[0], z[1]]), "LR": np.array([X0, y[1], z[1]])}
    return GroundTruth(windows={"r1c1": win}, plane_normal=np.array([-1.0, 0, 0]),
                       plane_d=X0, pose={}, lidar_bed=np.zeros(3))


def test_detects_floor_plane_below_panel():
    rng = np.random.default_rng(0)
    n = 2000
    floor = np.column_stack([rng.uniform(0.5, 2.5, n), rng.uniform(-0.5, 0.5, n),
                             1.20 + rng.normal(0, 0.01, n)])     # z ~ 1.20 (down)
    panel = np.column_stack([np.full(500, X0), rng.uniform(-0.5, 0.5, 500),
                             rng.uniform(-1.0, 0.0, 500)])       # above the floor
    pts = np.vstack([floor, panel])
    res = detect_floor(pts, make_gt())
    assert res is not None
    normal, d = res
    np.testing.assert_allclose(np.abs(normal), [0, 0, 1], atol=0.05)   # RANSAC fit, ~up
    assert abs(-d / normal[2] - 1.20) < 0.03                    # plane z ~ 1.20


def test_returns_none_without_enough_floor():
    rng = np.random.default_rng(1)
    panel = np.column_stack([np.full(300, X0), rng.uniform(-0.5, 0.5, 300),
                             rng.uniform(-1.0, 0.0, 300)])
    assert detect_floor(panel, make_gt()) is None
