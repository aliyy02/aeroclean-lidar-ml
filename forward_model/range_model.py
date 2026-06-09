"""Layer 1 -- the range equation, collapsed.

The full radar range equation for a beam-filling Lambertian target reduces (after the
illuminated-area R^2 cancels one R^2 of the two-way 1/R^4) to:

    P_r = C * rho_eff(theta) / R^n

  C        one lumped system constant (transmit power, gains, aperture, efficiencies,
           and the watts->count conversion). NOT separately identifiable from the
           per-material amplitudes -- fold it; only the product matters.
  rho_eff  the FULL Layer-3 angular curve I(theta). Already contains the diffuse cos.
  R        range (m).
  n        falloff exponent: 2 for beam-filling surfaces (wall, glass pane), drifting
           toward 4 for thin frame edges / small distant features. Fit per material.

THE TRAP: do not pass theta here and do not multiply another cos(theta) -- that is
the Lambertian incidence factor double-count. This function takes rho_eff and applies
only C/R^n, by construction.
"""
from __future__ import annotations

import numpy as np


def received_power(rho_eff, R, C: float, n: float = 2.0):
    """Received power P_r = C * rho_eff / R^n. Scalars or broadcastable arrays."""
    rho_eff = np.asarray(rho_eff, dtype=float)
    R = np.asarray(R, dtype=float)
    out = C * rho_eff / np.power(R, n)
    return out if out.ndim else float(out)
