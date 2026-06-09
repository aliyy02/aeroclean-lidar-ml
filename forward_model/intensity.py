"""Layer 3 -- the heart of the forward model.

Monostatic Lambertian + Cook-Torrance/Beckmann backscatter intensity
(Tian/Hyyppae et al., Sensors 2021), re-derived for the emitter=detector case:

    I(theta) = a*cos(theta) + b * [ exp(-tan^2(theta)/m^2) / cos^5(theta) ]

  a  diffuse weight   (~ rho/pi)      -- the matte floor, present at all angles
  b  specular weight  (~ Fresnel*G)   -- the near-normal mirror spike; INDEPENDENT of a
  m  roughness        (RMS facet slope) -- spike width; small m = sharp (glass), large = broad

This whole curve IS rho_eff(theta). The range step (Layer 1) applies only C/R^n on
top of it -- it must NOT multiply another cos(theta) (the diffuse cos is already here).
The cos^5 (not cos^4): the bare Beckmann distribution carries cos^4; the assembled
monostatic intensity picks up one more cos from the incidence projection.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def intensity(theta, a: float, b: float, m: float):
    """Layer-3 intensity I(theta) == rho_eff(theta). Scalar or array `theta` (radians).

    Numerically safe as theta -> pi/2: the specular term is forced to 0 where
    cos(theta) collapses (exp(-inf) would otherwise meet 1/0).
    """
    theta = np.asarray(theta, dtype=float)
    cos_t = np.cos(theta)
    diffuse = a * cos_t

    safe = cos_t > _EPS
    cos_safe = np.where(safe, cos_t, 1.0)          # avoid 0 in the divisions below
    tan2 = np.where(safe, (np.sin(theta) / cos_safe) ** 2, 0.0)
    spec = np.where(
        safe,
        b * np.exp(-tan2 / (m * m)) / cos_safe ** 5,
        0.0,
    )

    out = diffuse + spec
    return out if out.ndim else float(out)
