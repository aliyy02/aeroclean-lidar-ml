"""Per-point labeling on a synthetic scene with known geometry.

Scene (corrected NED, sensor at origin, facade ahead at +x):
  glass plane x0=1.645; one window y in [-0.5,0.5], z in [-1.0,0.0];
  floor plane z=1.0 (normal up).
"""
import numpy as np

from forward_model import materials as M
from calibration.gt_parse import GroundTruth
from calibration.label import label_points, LabelParams, INTERIOR, DROP

X0 = 1.645


def make_gt():
    y0, y1, z0, z1 = -0.5, 0.5, -1.0, 0.0
    win = {"UL": np.array([X0, y0, z0]), "UR": np.array([X0, y1, z0]),
           "LL": np.array([X0, y0, z1]), "LR": np.array([X0, y1, z1])}
    return GroundTruth(windows={"r1c1": win}, plane_normal=np.array([-1.0, 0, 0]),
                       plane_d=X0, pose={}, lidar_bed=np.zeros(3))


FLOOR = (np.array([0.0, 0.0, 1.0]), -1.0)    # z = 1.0 (down), normal up


def _label(pts, floor=FLOOR, params=None):
    return label_points(np.asarray(pts, float), make_gt(),
                        params or LabelParams(), floor=floor)


def test_glass_front_point():
    res = _label([[X0, 0.0, -0.5]])
    assert res.labels[0] == M.GLASS


def test_interior_see_through_point_with_beam_displacement():
    # beam through window center (X0,0,-0.5) continued DEEP (past the recessed pane) -> drifts
    scale = 2.1 / X0                            # 2.1 m: ~0.45 m behind the GT plane > interior_cut
    pt = [2.1, 0.0 * scale, -0.5 * scale]
    res = _label([pt])
    assert res.labels[0] == INTERIOR


def test_other_mullion_point_on_facade_just_outside_window():
    res = _label([[X0, 0.52, -0.5]])           # on plane, just past window edge, within margin
    assert res.labels[0] == M.NOT_GLASS


def test_point_beyond_grid_is_cropped():
    res = _label([[X0, 2.0, -0.5]])            # facade but far outside grid bbox
    assert res.labels[0] == DROP


def test_ground_point_on_floor_within_panel_y():
    res = _label([[1.0, 0.0, 1.0]])            # on floor plane z=1, y in panel extent
    assert res.labels[0] == M.GROUND_3


def test_floor_point_outside_panel_y_is_cropped():
    res = _label([[1.0, 3.0, 1.0]])            # on floor but y beyond panel -> drop
    assert res.labels[0] == DROP


def test_point_behind_sensor_is_cropped():
    res = _label([[-1.0, 0.0, -0.5]])          # x<=0, no valid facade beam
    assert res.labels[0] == DROP


def test_normals_facade_toward_sensor_ground_up():
    res = _label([[X0, 0.0, -0.5], [1.0, 0.0, 1.0]])   # glass, ground
    np.testing.assert_allclose(res.normals[0], [-1.0, 0.0, 0.0])   # facade -> toward sensor
    np.testing.assert_allclose(res.normals[1], [0.0, 0.0, -1.0])   # ground -> up


def test_returns_label_and_normal_per_point():
    pts = [[X0, 0.0, -0.5], [X0, 2.0, -0.5], [1.0, 0.0, 1.0]]
    res = _label(pts)
    assert res.labels.shape == (3,)
    assert res.normals.shape == (3, 3)
