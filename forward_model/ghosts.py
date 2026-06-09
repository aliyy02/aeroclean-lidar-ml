"""Optional Stage 2 -- ghost (mirror) points.

When a beam hits reflective/coated glass off-normal it can mirror-bounce to another
object and return along the bent path; the sensor then places a FALSE point at that
object's position mirrored across the glass plane. The reflection is a Householder
transform across the plane {x : n.x + d = 0}:

    p' = p - 2 (n.p + d) n        (n a unit normal)

Conditions for emitting ghosts (measured by Fong & Yan; applied by the orchestrator):
reflective/coated glass AND an occluder in front AND sensor height > 0.7 m AND
range <= 7.5 m. Clear float glass never ghosts. This module is just the geometry.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def mirror_across_plane(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Reflect each point across the plane n.x + d = 0. `normal` need not be unit length."""
    points = np.asarray(points, dtype=float)
    normal = np.asarray(normal, dtype=float)
    n_hat = normal / max(float(np.linalg.norm(normal)), _EPS)
    signed_dist = points @ n_hat + d           # (N,)
    return points - 2.0 * signed_dist[:, None] * n_hat[None, :]
