"""Tests for the Layer-3 monostatic Lambertian + Cook-Torrance/Beckmann intensity.

    I(theta) = a*cos(theta) + b*[ exp(-tan^2(theta)/m^2) / cos^5(theta) ]
"""
import numpy as np

from forward_model.intensity import intensity


def test_at_normal_incidence_intensity_is_a_plus_b():
    # theta=0: cos=1, tan=0, exp(0)=1, cos^5=1  ->  I = a + b
    assert np.isclose(intensity(0.0, a=0.3, b=0.7, m=0.1), 1.0)


def test_pure_diffuse_falls_as_cosine():
    # b=0 -> I = a*cos(theta)
    assert np.isclose(intensity(np.pi / 3, a=0.8, b=0.0, m=0.1), 0.8 * 0.5)


def test_narrower_roughness_decays_specular_faster():
    # Just off normal, the smaller-m (narrower) lobe must be dimmer.
    theta = 0.1
    narrow = intensity(theta, a=0.0, b=1.0, m=0.05)
    broad = intensity(theta, a=0.0, b=1.0, m=0.5)
    assert narrow < broad


def test_specular_vanishes_toward_grazing():
    val = intensity(1.5, a=0.0, b=1.0, m=0.1)  # ~85.9 deg
    assert 0.0 <= val < 1e-3


def test_numerically_safe_at_ninety_degrees():
    val = intensity(np.pi / 2, a=0.0, b=1.0, m=0.1)
    assert np.isfinite(val)
    assert np.isclose(val, 0.0)


def test_vectorized_over_theta():
    theta = np.array([0.0, np.pi / 4, np.pi / 2])
    out = intensity(theta, a=0.5, b=0.5, m=0.2)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    assert np.isclose(out[0], 1.0)            # a+b at normal
    assert np.isclose(out[2], 0.0)            # zero at grazing
