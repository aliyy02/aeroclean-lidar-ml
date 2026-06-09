"""Per-point labeling of a real scan against test_bed ground truth.

Input points are in the **corrected NED frame** (axis-remapped + de-rolled), sensor at the
origin. The facade is an arbitrary plane (the rig sweeps pose, so it is generally tilted):
`plane_normal . X + plane_d = 0`, normal pointing back toward the sensor. Each point is
classified by where its beam crosses that plane and by its signed distance to it:

  GLASS     : beam crosses inside a window rect, return near the plane (front shell)
  INTERIOR  : beam crosses inside a window rect, return BEHIND the plane (see-through)
  NOT_GLASS : on the plane, inside the window-grid box, but not in any window (mullion/frame)
  GROUND    : on the floor plane (found by RANSAC), within the panel's lateral extent
  DROP (-1) : everything else (room/clutter beyond the panel) -> cropped out

Labels 0/1/2 reuse forward_model.materials ids so real data shares the sim's label space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from forward_model import materials as M
from .gt_parse import GroundTruth

GLASS = M.GLASS            # 1
NOT_GLASS = M.NOT_GLASS    # 0
GROUND = M.GROUND_3        # 2
INTERIOR = 3               # real-scan-only: see-through return from behind the glass
DROP = -1                  # cropped / ignored


@dataclass
class LabelParams:
    tau_front: float = 0.05       # tolerance IN FRONT of the GT plane still counted as panel (m)
    interior_cut: float = 0.28    # in-window returns deeper than this behind the GT plane = interior
    interior_margin: float = 0.12  # when auto-estimated: interior_cut = panel_offset + this (m)
    grid_margin: float = 0.05     # window-grid bbox margin (m)
    tau_floor: float = 0.05       # floor-plane tolerance (m)


@dataclass
class LabelResult:
    labels: np.ndarray          # (N,) int
    normals: np.ndarray         # (N,3) float


def plane_basis(gt: GroundTruth) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unit normal + orthonormal in-plane axes (e_u along window width, e_v along height)
    + origin O (centroid of all window corners). All windows share orientation, so the
    first window's edges define the basis for the whole grid."""
    n = np.asarray(gt.plane_normal, float)
    n = n / np.linalg.norm(n)
    w0 = next(iter(gt.windows.values()))
    e_u = w0["UR"] - w0["UL"]
    e_u = e_u / np.linalg.norm(e_u)
    e_v = w0["LL"] - w0["UL"]
    e_v = e_v - (e_v @ e_u) * e_u          # orthogonalize against e_u
    e_v = e_v / np.linalg.norm(e_v)
    O = np.mean([c for w in gt.windows.values() for c in w.values()], axis=0)
    return n, e_u, e_v, O


def _window_rects_uv(gt, e_u, e_v, O) -> List[Tuple[float, float, float, float]]:
    rects = []
    for w in gt.windows.values():
        us = [(c - O) @ e_u for c in w.values()]
        vs = [(c - O) @ e_v for c in w.values()]
        rects.append((min(us), max(us), min(vs), max(vs)))
    return rects


def estimate_panel_offset(ned: np.ndarray, gt: GroundTruth, depth_band=(0.0, 0.5),
                          min_pts: int = 50) -> float:
    """Median depth (m) that in-window returns sit BEHIND the GT plane.

    The real panel is offset behind the measured GT plane by an accumulated
    ruler/tape measurement error (consistent across a bag). This returns that offset
    (the recessed glass pane's depth) so labeling can anchor on the actual panel. 0 if
    too few in-window returns."""
    ned = np.asarray(ned, dtype=float)
    n, e_u, e_v, O = plane_basis(gt)
    d = gt.plane_d
    s = ned @ n + d
    nP = ned @ n
    ahead = nP < -1e-9
    t = np.where(ahead, -d / np.where(ahead, nP, 1.0), 0.0)
    Q = ned * t[:, None]
    qu = (Q - O) @ e_u
    qv = (Q - O) @ e_v
    in_win = np.zeros(ned.shape[0], bool)
    for (u0, u1, v0, v1) in _window_rects_uv(gt, e_u, e_v, O):
        in_win |= (qu >= u0) & (qu <= u1) & (qv >= v0) & (qv <= v1)
    behind = -s
    sel = ahead & in_win & (behind > depth_band[0]) & (behind < depth_band[1])
    if int(sel.sum()) < min_pts:
        return 0.0
    return float(np.median(behind[sel]))


def detect_floor(ned: np.ndarray, gt: GroundTruth, tol: float = 0.04, min_pts: int = 200,
                 iters: int = 400, z_below: float = 0.3, seed: int = 0
                 ) -> Optional[Tuple[np.ndarray, float]]:
    """RANSAC the floor plane (robust to a tilted rig). Considers points off the facade,
    keeps the largest plane whose inlier centroid is clearly below the sensor (+z).
    Returns (unit normal toward sensor, d) with normal . X + d = 0, or None."""
    ned = np.asarray(ned, dtype=float)
    n_f = np.asarray(gt.plane_normal, float); n_f /= np.linalg.norm(n_f)
    s = ned @ n_f + gt.plane_d
    cand = ned[np.abs(s) > 0.30]                 # exclude the facade itself
    if cand.shape[0] < min_pts:
        return None
    rng = np.random.default_rng(seed)
    best_inl, best = 0, None
    for _ in range(iters):
        i = rng.choice(cand.shape[0], 3, replace=False)
        a, b, c = cand[i]
        nrm = np.cross(b - a, c - a)
        L = np.linalg.norm(nrm)
        if L < 1e-9:
            continue
        nrm = nrm / L
        d0 = -nrm @ a
        inl = np.abs(cand @ nrm + d0) < tol
        cnt = int(inl.sum())
        if cnt <= best_inl or cnt < min_pts:
            continue
        if cand[inl][:, 2].mean() < z_below:      # must sit below the sensor
            continue
        best_inl, best = cnt, (nrm, d0)
    if best is None:
        return None
    nrm, d0 = best
    if d0 < 0:                                    # orient normal toward the sensor
        nrm, d0 = -nrm, -d0
    return nrm, float(d0)


def label_points(ned: np.ndarray, gt: GroundTruth, params: LabelParams = LabelParams(),
                 floor: Optional[Tuple[np.ndarray, float]] = None) -> LabelResult:
    """Label corrected-NED points against the GT facade plane + windows."""
    ned = np.asarray(ned, dtype=float)
    nrows = ned.shape[0]
    labels = np.full(nrows, DROP, dtype=int)

    n, e_u, e_v, O = plane_basis(gt)
    d = gt.plane_d
    m = params.grid_margin

    s = ned @ n + d                              # signed dist: sensor>0, on-plane 0, behind<0
    nP = ned @ n
    ahead = nP < -1e-9                           # beam actually meets the facade in front
    t = np.where(ahead, -d / np.where(ahead, nP, 1.0), 0.0)
    Q = ned * t[:, None]                         # beam-plane intersection
    qu = (Q - O) @ e_u
    qv = (Q - O) @ e_v

    rects = _window_rects_uv(gt, e_u, e_v, O)
    in_win = np.zeros(nrows, bool)
    for (u0, u1, v0, v1) in rects:
        in_win |= (qu >= u0) & (qu <= u1) & (qv >= v0) & (qv <= v1)
    in_win &= ahead
    gu0 = min(r[0] for r in rects); gu1 = max(r[1] for r in rects)
    gv0 = min(r[2] for r in rects); gv1 = max(r[3] for r in rects)
    in_bbox = ahead & (qu >= gu0 - m) & (qu <= gu1 + m) & (qv >= gv0 - m) & (qv <= gv1 + m)

    # The real panel sits BEHIND the measured GT plane (accumulated measurement offset).
    # Panel zone = from tau_front in front of the GT plane to interior_cut behind it; only
    # returns deeper than interior_cut (true see-through) are interior.
    panel_zone = (s <= params.tau_front) & (s >= -params.interior_cut)
    labels[in_win & panel_zone] = GLASS
    labels[in_win & (s < -params.interior_cut)] = INTERIOR
    other = in_bbox & ~in_win & panel_zone & (labels == DROP)
    labels[other] = NOT_GLASS

    fn_toward_sensor = None
    if floor is not None:
        fn = np.asarray(floor[0], float); fd = float(floor[1])
        fu = (ned - O) @ e_u
        on_floor = ((np.abs(ned @ fn + fd) <= params.tau_floor) &
                    (fu >= gu0 - m) & (fu <= gu1 + m) & (labels == DROP))
        labels[on_floor] = GROUND
        fn_toward_sensor = fn if fd > 0 else -fn   # origin is on the +d side of the plane

    normals = np.full((nrows, 3), np.nan)
    facade = (labels == GLASS) | (labels == INTERIOR) | (labels == NOT_GLASS)
    normals[facade] = n
    if fn_toward_sensor is not None:
        normals[labels == GROUND] = fn_toward_sensor
    return LabelResult(labels=labels, normals=normals)
