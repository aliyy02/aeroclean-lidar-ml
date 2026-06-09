"""Labeling on a TILTED facade plane (non-zero rig pitch/yaw) — the general case.

Most real scans sweep pose, so the facade is not at constant x. A window is placed on a
pitched plane via an in-plane basis; labels must come out right using beam->plane
intersection + signed distance, not the x=const shortcut.
"""
import numpy as np

from calibration.gt_parse import GroundTruth
from calibration.label import label_points, LabelParams, INTERIOR, DROP
from forward_model import materials as M


def tilted_gt(pitch_deg=20.0, d=1.645, hw=0.6, hh=0.5):
    th = np.radians(pitch_deg)
    n = np.array([-np.cos(th), 0.0, -np.sin(th)])           # facade normal toward sensor
    e_u = np.cross(n, [0, 0, 1.0]); e_u /= np.linalg.norm(e_u)
    e_v = np.cross(n, e_u); e_v /= np.linalg.norm(e_v)
    C = -d * n                                              # window centre, on the plane
    win = {"UL": C - hw * e_u + hh * e_v, "UR": C + hw * e_u + hh * e_v,
           "LL": C - hw * e_u - hh * e_v, "LR": C + hw * e_u - hh * e_v}
    gt = GroundTruth(windows={"r1c1": win}, plane_normal=n, plane_d=d,
                     pose={}, lidar_bed=np.zeros(3))
    return gt, C, n, e_u, e_v


def test_glass_on_tilted_plane_and_normal():
    gt, C, n, eu, ev = tilted_gt()
    res = label_points(C[None, :], gt, LabelParams())
    assert res.labels[0] == M.GLASS
    np.testing.assert_allclose(res.normals[0], n, atol=1e-9)


def test_interior_behind_tilted_plane():
    gt, C, n, eu, ev = tilted_gt()
    res = label_points((1.3 * C)[None, :], gt, LabelParams())   # deep behind (>interior_cut), same beam
    assert res.labels[0] == INTERIOR


def test_other_on_tilted_plane_outside_window():
    gt, C, n, eu, ev = tilted_gt(hw=0.6)
    res = label_points((C + 0.63 * eu)[None, :], gt, LabelParams())  # on plane, just past edge
    assert res.labels[0] == M.NOT_GLASS


def test_crop_beyond_tilted_grid():
    gt, C, n, eu, ev = tilted_gt()
    res = label_points((C + 2.0 * eu)[None, :], gt, LabelParams())
    assert res.labels[0] == DROP


def test_axis_aligned_still_works_via_general_path():
    # pitch=0 reduces to the x-facing facade
    gt, C, n, eu, ev = tilted_gt(pitch_deg=0.0)
    res = label_points(C[None, :], gt, LabelParams())
    assert res.labels[0] == M.GLASS
