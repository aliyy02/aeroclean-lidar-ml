"""Tests for Layer-2 cone gate, Layer-4 threshold, and position jitter."""
import numpy as np

from forward_model.returns import cone_gate, passes_threshold, add_jitter


def test_cone_gate_keeps_only_near_normal():
    theta = np.array([0.0, 0.05, 0.2])
    mask = cone_gate(theta, half_width=0.1)
    np.testing.assert_array_equal(mask, [True, True, False])


def test_threshold_is_strict_greater_than():
    P_r = np.array([0.5, 1.0, 2.0])
    mask = passes_threshold(P_r, T=1.0)
    np.testing.assert_array_equal(mask, [False, False, True])


def test_zero_sigma_leaves_points_unchanged():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = add_jitter(pts, sigma=0.0, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(out, pts)


def test_jitter_is_deterministic_with_seeded_rng_and_preserves_shape():
    pts = np.zeros((1000, 3))
    a = add_jitter(pts, sigma=0.01, rng=np.random.default_rng(42))
    b = add_jitter(pts, sigma=0.01, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    assert a.shape == pts.shape
    # Empirical std should be close to the requested sigma.
    assert abs(a.std() - 0.01) < 0.002


def test_jitter_does_not_mutate_input():
    pts = np.ones((5, 3))
    _ = add_jitter(pts, sigma=0.1, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(pts, np.ones((5, 3)))
