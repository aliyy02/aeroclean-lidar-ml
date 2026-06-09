"""Per-point face normals (Option A) + the per-capture world->body rotation solve.

We never rely on AirSim's rotation convention. Each capture, the scene shows several
big flat surfaces whose true world normals we know (each wall/panel front, the ground).
We fit those normals in the lidar's body frame and Kabsch-solve the one rotation that
maps known(world) -> measured(body). With that rotation, every point gets the EXACT
normal of the box face it landed on -- front face or protrusion reveal alike.
"""
from __future__ import annotations

import numpy as np

from forward_model import materials as M


def face_normal_world(point_world, box) -> np.ndarray:
    """World outward normal of the box face nearest `point_world` (handles all 6 faces).

    The face is the one the point sits closest to the boundary of (smallest margin
    half-|local|). Using the margin -- not the ratio |local|/half -- is robust to the
    extreme aspect ratios here (a 2.5 cm-thin, 24 m-tall wall), where a ratio test would
    let the huge in-plane axis beat the thin front face for points near a vertical edge.
    """
    local = box.R.T @ (np.asarray(point_world, float) - box.center)
    axis = int(np.argmin(box.half - np.abs(local)))
    sign = 1.0 if local[axis] >= 0 else -1.0
    return sign * box.R[:, axis]


def sensor_facing_world_normal(box) -> np.ndarray:
    """The face normal that points toward an outside/overhead sensor (for rotation fitting)."""
    if box.cls == M.GROUND:
        return np.array([0.0, 0.0, -1.0])      # up (NED)
    return box.R[:, 0].copy()                  # outward front of a facade element


def plane_normal(points: np.ndarray) -> np.ndarray:
    """Best-fit plane normal of a point cluster, oriented toward the origin (the sensor)."""
    cen = points.mean(0)
    _, _, vt = np.linalg.svd(points - cen)
    n = vt[2]
    return -n if n @ cen > 0 else n


def solve_world_to_body(world_normals, body_normals) -> np.ndarray:
    """Kabsch rotation M mapping each world normal to its measured body normal."""
    P = np.asarray(world_normals, float); Q = np.asarray(body_normals, float)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def estimate_M(body_points, names, index, sensor_world=None, min_pts: int = 40,
               max_nonplanarity: float = 0.15):
    """Estimate world->body rotation M from the scan's large flat surfaces.

    Returns (M, n_surfaces). M is None if fewer than two non-parallel surfaces are
    usable (rotation then under-determined).
    """
    body_points = np.asarray(body_points, float)
    names = np.asarray(names)
    wn, bn = [], []
    for nm in set(names.tolist()):
        if nm not in index:
            continue
        pts = body_points[names == nm]
        if pts.shape[0] < min_pts:
            continue
        cen = pts.mean(0)
        _, s, _ = np.linalg.svd(pts - cen)
        if s[0] == 0 or s[2] / s[0] > max_nonplanarity:     # not flat enough
            continue
        bn.append(plane_normal(pts))
        wn.append(sensor_facing_world_normal(index[nm]))
    wn = np.array(wn)
    if len(wn) < 2 or np.linalg.matrix_rank(wn, tol=0.1) < 2:
        return None, len(wn)
    return solve_world_to_body(wn, np.array(bn)), len(wn)


def per_point_normals(body_points, names, index, M, sensor_world) -> np.ndarray:
    """Exact per-point body-frame normals via the box face each point hit.

    The face AXIS comes from geometry; the SIGN is oriented toward the sensor, because the
    lidar only ever hits the surface side that faces it (so the outward normal of the hit
    face must point back at the sensor). This makes the sign robust to small residual
    rotation error that would otherwise flip normals across thin (cm-scale) boxes.
    """
    body_points = np.asarray(body_points, float)
    Mbw = M.T                                   # body -> world
    out = np.zeros((len(body_points), 3))
    for i, (b, nm) in enumerate(zip(body_points, names)):
        box = index.get(nm)
        if box is None:                         # unknown actor: best-effort toward sensor
            out[i] = -b / max(np.linalg.norm(b), 1e-9)
            continue
        world = sensor_world + Mbw @ b
        n = M @ face_normal_world(world, box)
        out[i] = -n if (b @ n) > 0 else n       # orient toward the sensor (origin in body frame)
    return out
