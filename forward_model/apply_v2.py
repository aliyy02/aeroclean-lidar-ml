"""Revised forward-model orchestrator (per point).

Splits the two things the old model fused:
  - REPORTED INTENSITY = rho(theta)            -- range-independent reflectivity
  - DETECTION          = transmit x return-chance x noise-floor (the only place 1/R^n lives)

    rho   = reflectance(theta; a,g,s,b,m)                         (reflectance.py)
    keep  = rand < (1-tau)*return_probability(theta; p_floor,cone)  AND
            C*rho/R^n_detect > T                                  (detect.py)
    report intensity = rho ; xyz += Gaussian(sigma)

`mats` is one Material (broadcast) or a per-point list (material_library.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import ranges, incidence_angles
from .reflectance import reflectance
from .detect import return_probability, passes_noise_floor
from .material_library import Material

_FIELDS = ("a", "g", "s", "b", "m", "p_floor", "cone", "tau", "n_index", "train_label")


@dataclass
class Constants:
    C: float = 1.0          # lumped system gain (folds with T; only the ratio matters)
    T: float = 1e-4         # detector noise floor (randomized in diversification)
    n_detect: float = 2.0   # range falloff for the DETECTION test only
    sigma: float = 0.005    # position jitter (m)
    sat: float = 255.0      # reported-intensity ceiling: the L2 clips at 255 (metal/glazing saturate)


@dataclass
class ScanResult:
    xyz: np.ndarray         # (K,3) surviving points
    intensity: np.ndarray   # (K,) reported reflectivity rho
    labels: np.ndarray      # (K,) 3-class train label


def _param_arrays(mats, n: int):
    if isinstance(mats, Material):
        mats = [mats] * n
    return {f: np.array([getattr(mt, f) for mt in mats]) for f in _FIELDS}


def apply_scan(points, normals, mats, constants: Constants = Constants(),
               rng: np.random.Generator | None = None) -> ScanResult:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    n = points.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    if n == 0:
        return ScanResult(np.zeros((0, 3)), np.zeros((0,)), np.zeros((0,), dtype=int))

    P = _param_arrays(mats, n)
    R = ranges(points)
    theta = incidence_angles(points, normals)

    rho = reflectance(theta, P["a"], P["g"], P["s"], P["b"], P["m"], n_index=P["n_index"])
    surface_prob = (1.0 - P["tau"]) * return_probability(theta, P["p_floor"], P["cone"])
    keep = ((rng.random(n) < surface_prob) &
            passes_noise_floor(rho, R, constants.C, constants.T, constants.n_detect))

    xyz = points[keep]
    if constants.sigma > 0 and xyz.shape[0]:
        xyz = xyz + rng.normal(0.0, constants.sigma, xyz.shape)
    return ScanResult(xyz=xyz, intensity=np.minimum(rho[keep], constants.sat),
                      labels=P["train_label"][keep].astype(int))
