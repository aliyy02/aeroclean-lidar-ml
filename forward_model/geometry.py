"""Per-point geometry: range and incidence angle.

The LiDAR sits at the origin of the frame these points live in, so the beam
direction to a point is simply that point's unit vector. The incidence angle is
the angle between that beam and the surface normal at the hit. We use |cos(theta)|
so the result is invariant to whether the stored normal points inward or outward.

For SYNTHETIC scenes the normal comes from the scene manifest (exact). For the
real calibration scans it comes from the known window plane (the test-bed corners),
with PCA as a fallback -- but those normal sources live in the calibration package;
this module only consumes an (N,3) array of normals.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def ranges(points: np.ndarray) -> np.ndarray:
    """Euclidean distance from the sensor origin to each point. Shape (N,) for (N,3)."""
    return np.linalg.norm(points, axis=1)


def incidence_angles(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Incidence angle theta (radians) per point, in [0, pi/2].

    theta = arccos(|beam_hat . normal_hat|). Inputs need not be unit length.
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)

    beam_norm = np.linalg.norm(points, axis=1, keepdims=True)
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    beam_hat = points / np.maximum(beam_norm, _EPS)
    normal_hat = normals / np.maximum(normal_norm, _EPS)

    cos_theta = np.abs(np.sum(beam_hat * normal_hat, axis=1))
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    return np.arccos(cos_theta)
