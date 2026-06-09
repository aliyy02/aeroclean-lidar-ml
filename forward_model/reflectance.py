"""Revised angular reflectance — one family for every material.

The OLD model used a Lambertian `a*cos(theta)` diffuse term. Real Unitree-L2 data shows
EVERY material is flatter than cosine, and the reflective ones (glass, metal) LIFT toward
grazing. So the diffuse part gets a flatness knob, plus a Fresnel grazing-lift term, plus
the near-normal specular burst:

    rho(theta) = a*((1-g)*cos + g)  +  s*fresnel_grazing(theta)  +  b*exp(-tan^2/m^2)/cos^5

  a  diffuse scale        g  flatness (0 = cosine fade, 1 = perfectly flat)
  s  grazing-lift scale   b  near-normal specular burst   m  burst width (roughness)

This is the REPORTED reflectivity (range-independent); the 1/R^n lives only in detection.
"""
from __future__ import annotations

import numpy as np


def fresnel_grazing(theta, n_index: float = 1.5):
    """Grazing-lift shape: 0 at normal, rising to 1 toward grazing (unpolarized Fresnel
    reflectance of a dielectric, baseline-subtracted and normalized)."""
    theta = np.asarray(theta, dtype=float)
    cos1 = np.cos(theta)
    sin2 = np.clip(np.sin(theta) / n_index, -1.0, 1.0)
    cos2 = np.sqrt(np.maximum(1.0 - sin2 * sin2, 0.0))
    rs = ((cos1 - n_index * cos2) / (cos1 + n_index * cos2 + 1e-12)) ** 2
    rp = ((cos2 - n_index * cos1) / (cos2 + n_index * cos1 + 1e-12)) ** 2
    r = 0.5 * (rs + rp)
    r0 = ((1.0 - n_index) / (1.0 + n_index)) ** 2
    return np.clip((r - r0) / (1.0 - r0), 0.0, 1.0)


def reflectance(theta, a: float, g: float, s: float, b: float, m: float,
                n_index: float = 1.5):
    """Angular reflectivity rho(theta) for one material (see module docstring)."""
    theta = np.asarray(theta, dtype=float)
    cos = np.cos(theta)
    diffuse = a * ((1.0 - g) * cos + g)
    grazing = s * fresnel_grazing(theta, n_index)
    cc = np.clip(cos, 1e-6, 1.0)
    tan2 = (np.sin(theta) / cc) ** 2
    burst = b * np.exp(-tan2 / (m * m)) / cc ** 5
    return diffuse + grazing + burst
