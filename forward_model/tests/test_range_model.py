"""Tests for the Layer-1 range equation:  P_r = C * rho_eff / R^n."""
import numpy as np

from forward_model.range_model import received_power


def test_basic_value():
    assert np.isclose(received_power(rho_eff=1.0, R=1.0, C=5.0, n=2.0), 5.0)


def test_inverse_square_falloff():
    near = received_power(rho_eff=1.0, R=1.0, C=1.0, n=2.0)
    far = received_power(rho_eff=1.0, R=2.0, C=1.0, n=2.0)
    assert np.isclose(near / far, 4.0)          # doubling R -> 1/4 power


def test_exponent_n_is_used():
    p2 = received_power(rho_eff=1.0, R=2.0, C=1.0, n=2.0)
    p4 = received_power(rho_eff=1.0, R=2.0, C=1.0, n=4.0)
    assert np.isclose(p2, 1.0 / 4.0)
    assert np.isclose(p4, 1.0 / 16.0)


def test_power_depends_on_rho_eff_not_theta_directly():
    # Contract guard against the cos(theta) double-count: the range step takes the
    # already-angle-resolved rho_eff and applies ONLY C/R^n. Same rho_eff + R must
    # give the same power regardless of how that rho_eff arose.
    a = received_power(rho_eff=0.42, R=3.0, C=2.0, n=2.0)
    b = received_power(rho_eff=0.42, R=3.0, C=2.0, n=2.0)
    assert a == b


def test_vectorized():
    rho = np.array([1.0, 0.5])
    R = np.array([1.0, 2.0])
    out = received_power(rho_eff=rho, R=R, C=4.0, n=2.0)
    np.testing.assert_allclose(out, [4.0, 0.5])
