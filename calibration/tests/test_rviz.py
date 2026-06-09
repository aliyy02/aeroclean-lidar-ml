"""Pure helpers for the RViz publisher: label->color and NED->view-frame."""
import numpy as np

from calibration.rviz_publish import label_to_rgb, ned_to_view
from calibration.label import NOT_GLASS, GLASS, GROUND, INTERIOR


def test_label_to_rgb_distinct_per_class():
    rgb = label_to_rgb(np.array([NOT_GLASS, GLASS, GROUND, INTERIOR]))
    assert rgb.shape == (4, 3) and rgb.dtype == np.uint8
    # all four classes get distinct colors
    assert len({tuple(c) for c in rgb}) == 4


def test_label_to_rgb_glass_is_cyan_ish():
    rgb = label_to_rgb(np.array([GLASS]))[0]
    assert rgb[2] > 150 and rgb[1] > 120 and rgb[0] < 100   # high blue+green, low red


def test_ned_to_view_maps_right_forward_up():
    # NED (x fwd, y right, z down) -> view (X right, Y forward, Z up)
    out = ned_to_view(np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(out, [[2.0, 1.0, -3.0]])


def test_ned_to_view_up_is_negative_ned_z():
    out = ned_to_view(np.array([[0.0, 0.0, -5.0]]))   # ned z=-5 is UP
    assert out[0, 2] == 5.0
