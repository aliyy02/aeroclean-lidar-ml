"""End-to-end on cached REAL L2 frames (offline): native -> corrected -> label.

Fixture = 3 frames of L6_test_1 (native xyz+intensity) + the GT .txt, cached so this
runs without ROS or the simulator.
"""
import os

import numpy as np
import pytest

from calibration.gt_parse import load_gt
from calibration.frames import native_to_ned, to_corrected
from calibration.roll_estimate import estimate_roll
from calibration.label import label_points, detect_floor, LabelParams, INTERIOR
from forward_model import materials as M

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "fixtures", "real_frames.npz")
GT = os.path.join(HERE, "fixtures", "L6_gt.txt")
ALPHA = 24.0


@pytest.fixture(scope="module")
def data():
    d = np.load(FIX)
    frames = [d[f"f{k}_xyz"].astype(float) for k in range(3)]
    inten = [d[f"f{k}_int"].astype(float) for k in range(3)]
    gt = load_gt(GT)
    return frames, inten, gt


def test_gt_has_eight_windows(data):
    _, _, gt = data
    assert len(gt.windows) == 8


def test_estimate_roll_runs_on_real_and_is_in_range(data):
    # NOTE: on a glass-filled panel embedded in a larger coplanar wall, the auto
    # estimate is only a rough initializer (the window grid's boundary is masked by
    # the surrounding wall). Visual tuning of alpha via the QA overlay is the reliable
    # path. Here we only assert the helper runs and returns a finite, in-range value.
    frames, _, gt = data
    agg = native_to_ned(np.vstack(frames))
    alpha = estimate_roll(agg, gt)
    assert np.isfinite(alpha) and -45.0 <= alpha <= 45.0


def test_classes_present_and_valid(data):
    frames, _, gt = data
    agg = to_corrected(np.vstack(frames), ALPHA)
    floor = detect_floor(agg, gt)
    res = label_points(agg, gt, LabelParams(), floor=floor)
    valid = {M.NOT_GLASS, M.GLASS, M.GROUND_3, INTERIOR, -1}
    assert set(np.unique(res.labels)).issubset(valid)
    assert (res.labels == M.GLASS).sum() > 0
    assert (res.labels == INTERIOR).sum() > 0
    assert (res.labels == M.NOT_GLASS).sum() > 0


def test_interior_is_behind_glass(data):
    frames, _, gt = data
    agg = to_corrected(np.vstack(frames), ALPHA)
    res = label_points(agg, gt, LabelParams(), floor=detect_floor(agg, gt))
    x = agg[:, 0]
    glass_x = x[res.labels == M.GLASS].mean()
    interior_x = x[res.labels == INTERIOR].mean()
    assert interior_x > glass_x + 0.05         # interior (deep see-through) sits behind the pane
    # the recessed pane is the bulk -> glass dominates; interior is only the deep tail
    assert (res.labels == M.GLASS).sum() > (res.labels == INTERIOR).sum()


def test_room_clutter_is_cropped(data):
    frames, _, gt = data
    agg = to_corrected(np.vstack(frames), ALPHA)
    res = label_points(agg, gt, LabelParams(), floor=detect_floor(agg, gt))
    assert (res.labels == -1).sum() > 0        # points beyond the panel dropped


def test_normals_assigned(data):
    frames, _, gt = data
    agg = to_corrected(np.vstack(frames), ALPHA)
    res = label_points(agg, gt, LabelParams(), floor=detect_floor(agg, gt))
    facade = (res.labels == M.GLASS) | (res.labels == INTERIOR) | (res.labels == M.NOT_GLASS)
    np.testing.assert_allclose(res.normals[facade][0], [-1.0, 0.0, 0.0])
    if (res.labels == M.GROUND_3).any():
        gn = res.normals[res.labels == M.GROUND_3][0]    # RANSAC-fit, ~up; not exact
        assert gn[2] < -0.9 and abs(gn[0]) < 0.3 and abs(gn[1]) < 0.3
