"""Estimate the boresight roll correction that levels a scan's panel to the GT grid.

The correction `alpha` (to feed to `frames.apply_roll`/`to_corrected`) is the angle that
makes the panel's edges axis-aligned. We find it by minimizing the area of the de-rolled
panel's axis-aligned bounding box over a sweep.

ACCURACY CAVEAT: this is a ROUGH INITIALIZER, reliable only when the window panel's outer
boundary is clean. On a glass-filled panel embedded in a larger coplanar wall (the real
L6 test bed), the grid boundary is masked by surrounding wall returns and the estimate is
untrustworthy -- set `alpha` from the visual QA overlay instead (label_scan --qa). The
synthetic unit tests pin the function's correctness on clean input.
"""
from __future__ import annotations

import numpy as np

from .gt_parse import GroundTruth


def estimate_roll(ned: np.ndarray, gt: GroundTruth,
                  search_deg=(-44.0, 44.0), step_deg: float = 0.25,
                  depth_band=(-0.15, 0.35), gate=(1.5, 1.5, 0.5),
                  pct=(1.0, 99.0)) -> float:
    """Return the roll correction (deg) that levels the panel.

    `ned` are body-NED points (axis-remapped, pre-roll). Selection: a depth band around
    the facade (front + see-through layers) AND a (y,z) window around the GT grid
    (`gate` = y-margin, z-up-margin, z-down-margin) that excludes the floor below and
    walls to the side -- these horizontal/side surfaces also cross the facade depth and
    would corrupt the boundary fit. Kept points are back-projected to the glass plane
    (removes the see-through radial spread) before the min-area-rectangle sweep.
    """
    ned = np.asarray(ned, dtype=float)
    x, y, z = ned[:, 0], ned[:, 1], ned[:, 2]
    X0 = gt.plane_x
    (gy0, gy1), (gz0, gz1) = gt.grid_bbox()
    my, mz_up, mz_dn = gate
    sel = ((x > X0 + depth_band[0]) & (x < X0 + depth_band[1]) &
           (y > gy0 - my) & (y < gy1 + my) &
           (z > gz0 - mz_up) & (z < gz1 + mz_dn))      # z+ is down: mz_dn excludes floor
    if sel.sum() < 10:
        raise ValueError("too few facade points to estimate roll")
    # back-project to glass plane (removes see-through radial spread)
    scale = X0 / x[sel]
    qy, qz = y[sel] * scale, z[sel] * scale

    angles = np.arange(search_deg[0], search_deg[1] + 1e-9, step_deg)
    lo, hi = pct
    best_alpha, best_area = 0.0, np.inf
    for a in angles:
        r = np.radians(a)
        c, s = np.cos(r), np.sin(r)
        uy = qy * c - qz * s          # R_x(a) acting on (y,z)
        uz = qy * s + qz * c
        area = ((np.percentile(uy, hi) - np.percentile(uy, lo)) *
                (np.percentile(uz, hi) - np.percentile(uz, lo)))
        if area < best_area:
            best_area, best_alpha = area, float(a)
    return best_alpha
