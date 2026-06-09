"""Frame transforms for real Unitree-L2 scans.

native L2  --native_to_ned-->  body-NED  --apply_roll(alpha)-->  corrected NED (== GT frame)

- The axis remap `ned = (sz, sx, sy)` was brute-force-confirmed against the data
  (forward = native z; differs from the sim's `_des_to_ned`).
- The roll is the mounting-hole-vs-true-axes offset about the boresight (ned_x) that
  `test_bed.py` does not model. `alpha` is a calibration constant (~24 deg), tunable.
"""
from __future__ import annotations

import numpy as np

# native column order is (x, y, z); we want (z, x, y)
_REMAP = [2, 0, 1]


def native_to_ned(xyz_native: np.ndarray) -> np.ndarray:
    """Map native L2 points (...,3) to body-NED: ned_x=sz, ned_y=sx, ned_z=sy."""
    xyz_native = np.asarray(xyz_native, dtype=float)
    return xyz_native[..., _REMAP]


def roll_matrix(alpha_deg: float) -> np.ndarray:
    """Right-handed rotation about the boresight (x) axis by alpha (degrees)."""
    a = np.radians(alpha_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def apply_roll(ned: np.ndarray, alpha_deg: float) -> np.ndarray:
    """Rotate body-NED points (...,3) about x by alpha. Leaves ned_x unchanged."""
    ned = np.asarray(ned, dtype=float)
    return ned @ roll_matrix(alpha_deg).T


def to_corrected(xyz_native: np.ndarray, alpha_deg: float) -> np.ndarray:
    """Full transform: native L2 -> corrected NED (axis remap, then de-roll)."""
    return apply_roll(native_to_ned(xyz_native), alpha_deg)
