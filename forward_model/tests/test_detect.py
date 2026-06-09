"""Detection split: a soft angle-dependent return CHANCE (replaces the hard cone) and a
noise-floor test on the RAW power (the only place 1/R^n lives now)."""
import numpy as np

from forward_model.detect import return_probability, passes_noise_floor


def test_diffuse_returns_at_all_angles():
    # p_floor = 1 -> always returns regardless of angle (coated glass / matte)
    th = np.radians([0, 30, 60, 85])
    np.testing.assert_allclose(return_probability(th, p_floor=1.0, cone=0.1), 1.0)


def test_specular_returns_only_near_normal():
    # p_floor = 0, narrow cone -> ~1 at normal, ~0 off-normal (mirror glass)
    assert return_probability(0.0, p_floor=0.0, cone=np.radians(5)) > 0.99
    assert return_probability(np.radians(30), p_floor=0.0, cone=np.radians(5)) < 0.01


def test_partial_floor_is_between():
    # p_floor=0.3 -> off-normal chance floors at 0.3, boosted near normal
    far = return_probability(np.radians(60), p_floor=0.3, cone=np.radians(5))
    near = return_probability(0.0, p_floor=0.3, cone=np.radians(5))
    assert abs(far - 0.3) < 1e-6
    assert near > 0.99


def test_noise_floor_drops_far_faint_returns():
    # same reflectivity, near vs far: raw power falls as 1/R^2
    rho = np.array([0.5, 0.5])
    R = np.array([2.0, 40.0])
    keep = passes_noise_floor(rho, R, C=1.0, T=1e-3, n=2.0)
    assert keep[0] and not keep[1]


def test_noise_floor_uses_raw_power_not_reported():
    # a bright-but-far return can still fall below the floor
    assert passes_noise_floor(np.array([1.0]), np.array([1.0]), C=1.0, T=0.1, n=2.0)[0]
    assert not passes_noise_floor(np.array([1.0]), np.array([100.0]), C=1.0, T=0.1, n=2.0)[0]
