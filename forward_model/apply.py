"""Per-point forward-model orchestrator: geometry -> intensity -> range -> gate -> noise.

Input is a raw Standard-Lidar scan in the frame where the normals live (sensor at the
origin): xyz, the per-point actor names (groundtruth strings), and a per-point surface
normal (resolved from the scene manifest by the caller). Output is the surviving points
with synthesized intensity (= received power) and a class label per point.

This is the only place the layers compose, so the cos(theta) single-count rule lives
here by construction: rho_eff = I(theta) (Layer 3) already holds the diffuse cosine;
the range step applies only C/R^n (Layer 1) -- never a second cosine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import materials as M
from .constants import ForwardModelConstants
from .geometry import ranges, incidence_angles
from .intensity import intensity
from .range_model import received_power
from .returns import cone_gate, passes_threshold, add_jitter

# Ids in a fixed order so we can build per-id lookup arrays.
_IDS = (M.WALL, M.METAL_FRAME, M.GLASS_CLEAR, M.GLASS_COATED,
        M.GLASS_LOWE, M.SPANDREL, M.GROUND, M.OTHER)


@dataclass
class ScanResult:
    xyz: np.ndarray         # (K,3) surviving points, jittered
    intensity: np.ndarray   # (K,) received power (proportional to reported LiDAR count)
    labels: np.ndarray      # (K,) class id per point


def _per_id_array(constants: ForwardModelConstants, attr: str, dtype=float) -> np.ndarray:
    """Lookup array indexed by class id (0..7) for a MaterialParams attribute."""
    out = np.zeros(len(_IDS), dtype=dtype)
    for mid in _IDS:
        out[mid] = getattr(constants.materials[mid], attr)
    return out


def apply_forward_model(points, names, normals, constants: ForwardModelConstants,
                        rng: np.random.Generator | None = None) -> ScanResult:
    """Run the 4-layer forward model over a scan. Returns surviving points + intensity + labels."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)

    if points.shape[0] == 0:
        return ScanResult(np.zeros((0, 3)), np.zeros((0,)), np.zeros((0,), dtype=int))

    ids = np.array([M.material_for(n) for n in names], dtype=int)

    # Gather per-point material parameters via id-indexed lookup arrays.
    a = _per_id_array(constants, "a")[ids]
    b = _per_id_array(constants, "b")[ids]
    m = _per_id_array(constants, "m")[ids]
    n_exp = _per_id_array(constants, "n")[ids]
    is_glass = _per_id_array(constants, "is_glass", dtype=bool)[ids]
    cone = _per_id_array(constants, "cone_half_width")[ids]

    R = ranges(points)
    theta = incidence_angles(points, normals)

    rho_eff = intensity(theta, a, b, m)                 # Layer 3 (== rho_eff)
    P_r = received_power(rho_eff, R, constants.C, n_exp)  # Layer 1 (C/R^n only)

    keep = passes_threshold(P_r, constants.T)            # Layer 4
    # Layer 2: glass points must also lie inside their near-normal cone.
    keep &= ~is_glass | cone_gate(theta, cone)

    xyz_keep = add_jitter(points[keep], constants.sigma, rng)  # Layer 4 jitter
    return ScanResult(xyz=xyz_keep, intensity=P_r[keep], labels=ids[keep])
