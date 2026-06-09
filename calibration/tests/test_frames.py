"""Native L2 -> body-NED axis remap, and the boresight roll correction."""
import numpy as np

from calibration.frames import native_to_ned, roll_matrix, apply_roll, to_corrected


def test_native_to_ned_axis_remap():
    # native (sx,sy,sz) -> ned (sz, sx, sy):  ned_x=fwd=sz, ned_y=right=sx, ned_z=down=sy
    out = native_to_ned(np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(out, [[3.0, 1.0, 2.0]])


def test_native_forward_becomes_ned_x():
    # a far-forward native point (big sz) must land at big ned_x
    out = native_to_ned(np.array([[0.1, 0.2, 5.0]]))
    assert out[0, 0] == 5.0


def test_roll_matrix_identity_and_90():
    np.testing.assert_allclose(roll_matrix(0.0), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        roll_matrix(90.0), [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-9)


def test_roll_preserves_boresight_x():
    pts = np.array([[1.645, 2.0, -3.0], [1.7, -1.0, 0.5]])
    out = apply_roll(pts, 24.0)
    np.testing.assert_allclose(out[:, 0], pts[:, 0])      # x unchanged by roll about x


def test_roll_rotates_yz():
    out = apply_roll(np.array([[1.6, 1.0, 0.0]]), 90.0)
    np.testing.assert_allclose(out, [[1.6, 0.0, 1.0]], atol=1e-9)


def test_roll_round_trip():
    pts = np.array([[1.6, 2.3, -1.1], [2.0, -0.5, 0.7]])
    np.testing.assert_allclose(apply_roll(apply_roll(pts, 24.0), -24.0), pts, atol=1e-12)


def test_to_corrected_composes_remap_then_roll():
    native = np.array([[1.0, 2.0, 3.0]])
    expect = apply_roll(native_to_ned(native), 24.0)
    np.testing.assert_allclose(to_corrected(native, 24.0), expect)
