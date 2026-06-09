"""Layer 2 (glass cone gate) and Layer 4 (threshold + position jitter).

Layer 2: a monostatic sensor only catches a specular glass return inside a narrow
near-normal cone whose half-width equals the Beckmann roughness m. Outside the cone
the mirror bounce misses the detector -> no direct return (optionally a ghost).

Layer 4: a beam is recorded iff its received power clears the detector noise floor T.
Surviving points get a small Gaussian position jitter (measured on flat real surfaces).
"""
from __future__ import annotations

import numpy as np


def cone_gate(theta, half_width: float) -> np.ndarray:
    """Boolean mask: True where theta <= the cone half-width (glass direct-return zone)."""
    return np.asarray(theta, dtype=float) <= half_width


def passes_threshold(P_r, T: float) -> np.ndarray:
    """Boolean mask: True where received power strictly exceeds the noise floor T."""
    return np.asarray(P_r, dtype=float) > T


def add_jitter(points: np.ndarray, sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a copy of points with i.i.d. Gaussian(0, sigma) added to each coordinate."""
    points = np.asarray(points, dtype=float)
    if sigma == 0.0:
        return points.copy()
    if rng is None:
        rng = np.random.default_rng()
    return points + rng.normal(0.0, sigma, size=points.shape)
