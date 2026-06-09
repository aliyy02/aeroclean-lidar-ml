"""Detection: does a beam come back at all? Two independent gates.

(1) return CHANCE vs angle -- a soft replacement for the old hard glass cone:
      P_return(theta) = p_floor + (1 - p_floor) * exp(-theta^2 / (2*cone^2))
    p_floor = 1 -> matte/coated (returns at every angle); p_floor ~ 0 -> mirror (near-normal only).

(2) noise floor on the RAW received power (this is the ONLY place 1/R^n lives now -- the
    reported intensity is range-independent):
      keep iff  C * rho(theta) / R^n  >  T
"""
from __future__ import annotations

import numpy as np


def return_probability(theta, p_floor: float, cone: float):
    """Chance a beam returns, given incidence angle (radians) and material gate params."""
    theta = np.asarray(theta, dtype=float)
    boost = np.exp(-(theta * theta) / (2.0 * cone * cone))
    return p_floor + (1.0 - p_floor) * boost


def passes_noise_floor(rho, R, C: float, T: float, n: float = 2.0):
    """True where raw received power C*rho/R^n clears the detector noise floor T."""
    rho = np.asarray(rho, dtype=float)
    R = np.asarray(R, dtype=float)
    raw_power = C * rho / np.power(R, n)
    return raw_power > T
