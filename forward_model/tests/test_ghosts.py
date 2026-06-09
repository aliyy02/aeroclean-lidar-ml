"""Tests for the optional ghost (mirror) point stage: Householder reflection."""
import numpy as np

from forward_model.ghosts import mirror_across_plane


def test_reflect_across_yz_plane():
    # Plane x = 0: normal (1,0,0), d = 0. (3,1,2) -> (-3,1,2).
    p = np.array([[3.0, 1.0, 2.0]])
    out = mirror_across_plane(p, normal=np.array([1.0, 0.0, 0.0]), d=0.0)
    np.testing.assert_allclose(out, [[-3.0, 1.0, 2.0]])


def test_reflect_across_offset_plane():
    # Plane x = 5: normal (1,0,0), d = -5. (3,1,2) mirrors to (7,1,2).
    p = np.array([[3.0, 1.0, 2.0]])
    out = mirror_across_plane(p, normal=np.array([1.0, 0.0, 0.0]), d=-5.0)
    np.testing.assert_allclose(out, [[7.0, 1.0, 2.0]])


def test_normalizes_nonunit_normal():
    p = np.array([[3.0, 1.0, 2.0]])
    out = mirror_across_plane(p, normal=np.array([2.0, 0.0, 0.0]), d=0.0)  # length 2
    np.testing.assert_allclose(out, [[-3.0, 1.0, 2.0]])


def test_point_on_plane_is_unchanged():
    p = np.array([[0.0, 4.0, 9.0]])  # lies on x=0
    out = mirror_across_plane(p, normal=np.array([1.0, 0.0, 0.0]), d=0.0)
    np.testing.assert_allclose(out, p, atol=1e-12)


def test_vectorized():
    pts = np.array([[3.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    out = mirror_across_plane(pts, normal=np.array([1.0, 0.0, 0.0]), d=0.0)
    np.testing.assert_allclose(out, [[-3.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
