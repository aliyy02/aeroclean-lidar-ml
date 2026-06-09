"""Revised angular reflectance rho(theta) — one family spanning all materials.

rho(theta) = a*((1-g)*cos + g)  +  s*fresnel_grazing(theta)  +  b*beckmann_burst(theta, m)
  a = diffuse scale,  g = flatness (0=cosine fade, 1=flat),
  s = grazing-lift scale,  b/m = near-normal specular burst.
This is the reported (range-independent) reflectivity; no 1/R^n here.
"""
import numpy as np

from forward_model.reflectance import reflectance, fresnel_grazing


def test_lambertian_limit_is_cosine():
    th = np.radians([0, 30, 60])
    out = reflectance(th, a=1.0, g=0.0, s=0.0, b=0.0, m=0.1)
    np.testing.assert_allclose(out, np.cos(th), atol=1e-9)


def test_flat_limit_is_constant():
    th = np.radians([0, 30, 60, 80])
    out = reflectance(th, a=1.0, g=1.0, s=0.0, b=0.0, m=0.1)
    np.testing.assert_allclose(out, 1.0, atol=1e-9)


def test_fresnel_grazing_zero_at_normal_rises_to_one_at_grazing():
    assert abs(fresnel_grazing(0.0)) < 1e-6
    assert fresnel_grazing(np.radians(85)) > 0.4     # meaningful lift by 85 deg
    assert fresnel_grazing(np.radians(89.5)) > 0.8   # approaches 1 only near grazing
    # monotonically increasing (unpolarized external Fresnel)
    th = np.radians(np.linspace(0, 89, 50))
    f = fresnel_grazing(th)
    assert np.all(np.diff(f) >= -1e-9)


def test_grazing_term_lifts_at_high_angle():
    # pure grazing term: brighter at 70deg than at 10deg
    lo = reflectance(np.radians(10), a=0, g=0, s=1.0, b=0, m=0.1)
    hi = reflectance(np.radians(70), a=0, g=0, s=1.0, b=0, m=0.1)
    assert hi > lo


def test_specular_burst_peaks_at_normal_and_collapses():
    peak = reflectance(np.array([0.0]), a=0, g=0, s=0, b=1.0, m=0.05)[0]
    off = reflectance(np.array([np.radians(15)]), a=0, g=0, s=0, b=1.0, m=0.05)[0]
    assert peak > 0.9
    assert off < 0.05 * peak          # narrow burst gone within 15 deg


def test_all_terms_add():
    th = np.radians(20)
    expect = (reflectance(th, a=1, g=0.3, s=0, b=0, m=0.1)
              + reflectance(th, a=0, g=0, s=0.5, b=0, m=0.1)
              + reflectance(th, a=0, g=0, s=0, b=0.4, m=0.1))
    got = reflectance(th, a=1, g=0.3, s=0.5, b=0.4, m=0.1)
    np.testing.assert_allclose(got, expect, atol=1e-9)


def test_stays_finite_near_grazing():
    out = reflectance(np.radians([88, 89, 89.9]), a=0.5, g=0.5, s=0.5, b=0.5, m=0.1)
    assert np.all(np.isfinite(out))
