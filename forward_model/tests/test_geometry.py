"""Tests for per-point range and incidence-angle geometry."""
import numpy as np

from forward_model.geometry import ranges, incidence_angles


def test_ranges_are_euclidean_norms():
    pts = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 12.0]])
    np.testing.assert_allclose(ranges(pts), [3.0, 4.0, 12.0])


def test_incidence_zero_when_beam_hits_along_normal():
    # Point straight ahead; surface normal facing back toward the sensor.
    pts = np.array([[5.0, 0.0, 0.0]])
    normals = np.array([[-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(incidence_angles(pts, normals), [0.0], atol=1e-7)


def test_incidence_is_normal_sign_invariant():
    # Outward-pointing normal gives the same incidence angle (we use |cos|).
    pts = np.array([[5.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        incidence_angles(pts, np.array([[1.0, 0.0, 0.0]])),
        incidence_angles(pts, np.array([[-1.0, 0.0, 0.0]])),
        atol=1e-7,
    )


def test_incidence_forty_five_degrees():
    pts = np.array([[1.0, 1.0, 0.0]])          # beam at 45 deg in the XY plane
    normals = np.array([[-1.0, 0.0, 0.0]])     # facade normal along X
    np.testing.assert_allclose(
        incidence_angles(pts, normals), [np.pi / 4], atol=1e-6
    )


def test_incidence_grazing_is_ninety_degrees():
    pts = np.array([[0.0, 1.0, 0.0]])          # beam perpendicular to the normal
    normals = np.array([[-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        incidence_angles(pts, normals), [np.pi / 2], atol=1e-6
    )


def test_incidence_normalizes_unnormalized_normals():
    pts = np.array([[1.0, 1.0, 0.0]])
    np.testing.assert_allclose(
        incidence_angles(pts, np.array([[-2.0, 0.0, 0.0]])),  # length 2, not unit
        [np.pi / 4],
        atol=1e-6,
    )
