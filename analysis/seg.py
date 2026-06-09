"""Reusable segmentation + per-point measurement core for real L2 facade scans.

Frontal-facade physics quantities (incidence angle, range, depth-behind-plane,
intensity) are INVARIANT to the boresight roll about x, so the roll only matters
for window-vs-frame aperture assignment. This module:

  - axis-maps native L2 -> NED as ned = (z, ysign*x, y)   [z-depth confirmed]
  - de-rolls about x by alpha
  - classifies each beam by where it crosses the GT facade plane:
      'glass'  = inside a GT window aperture
      'frame'  = inside the window-grid bbox but not in any window (mullion/wall)
      'off'    = outside the grid bbox
  - fits the real facade FRONT plane robustly (to the on-frame returns) so depth
    and incidence angle can be referenced to the true surface, not just GT.
  - returns per-point: aperture class, range R, incidence angle theta (vs GT and
    vs fitted normal), depth behind GT plane, depth behind fitted front, intensity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, ".")
from calibration.io_bag import read_all                  # noqa: E402
from calibration.gt_parse import GroundTruth, load_gt    # noqa: E402
from calibration.frames import apply_roll                 # noqa: E402

GLASS, FRAME, OFF = 1, 0, -1


def axis_map(xyz: np.ndarray, ysign: int = 1) -> np.ndarray:
    out = np.empty_like(xyz)
    out[:, 0] = xyz[:, 2]
    out[:, 1] = ysign * xyz[:, 0]
    out[:, 2] = xyz[:, 1]
    return out


def plane_basis(gt: GroundTruth):
    n = np.asarray(gt.plane_normal, float); n = n / np.linalg.norm(n)
    w0 = next(iter(gt.windows.values()))
    e_u = w0["UR"] - w0["UL"]; e_u = e_u / np.linalg.norm(e_u)
    e_v = w0["LL"] - w0["UL"]; e_v = e_v - (e_v @ e_u) * e_u; e_v = e_v / np.linalg.norm(e_v)
    O = np.mean([c for w in gt.windows.values() for c in w.values()], axis=0)
    return n, e_u, e_v, O


def window_rects_uv(gt, e_u, e_v, O):
    rects = []
    for w in gt.windows.values():
        us = [(c - O) @ e_u for c in w.values()]
        vs = [(c - O) @ e_v for c in w.values()]
        rects.append((min(us), max(us), min(vs), max(vs)))
    return rects


@dataclass
class Seg:
    ned: np.ndarray          # (N,3) corrected NED
    intensity: np.ndarray    # (N,)
    cls: np.ndarray          # (N,) GLASS/FRAME/OFF (aperture class)
    R: np.ndarray            # (N,) range
    theta_gt: np.ndarray     # (N,) incidence vs GT normal (rad)
    behind_gt: np.ndarray    # (N,) depth behind GT plane (m, +behind)
    theta_fit: np.ndarray    # (N,) incidence vs fitted front normal (rad)
    behind_fit: np.ndarray   # (N,) depth behind fitted front plane (m, +behind)
    u: np.ndarray            # (N,) in-plane horiz (beam->GT-plane)
    v: np.ndarray            # (N,) in-plane vert
    fit_normal: np.ndarray   # (3,) fitted front-plane unit normal toward sensor
    fit_d: float             # fitted plane: n.X + d = 0
    frame_id: np.ndarray     # (N,) source frame index


def _fit_front_plane(ned, cls, behind_gt, n_gt, grid_margin=0.05):
    """Robust front-surface plane from on-frame returns near the GT plane.
    Falls back to GT normal if too few points."""
    sel = (cls == FRAME) & (behind_gt > -0.10) & (behind_gt < 0.35)
    P = ned[sel]
    if P.shape[0] < 200:
        # fall back: densest facade layer overall
        sel = (cls != OFF) & (behind_gt > -0.1) & (behind_gt < 0.5)
        P = ned[sel]
        if P.shape[0] < 50:
            return n_gt, None
    # take the FRONT shell: points within 6 cm of the per-point frontmost layer
    # robust: fit plane by PCA on the front 40% by depth (closest to sensor)
    order = np.argsort((P @ n_gt))            # n_gt points to sensor; larger = closer
    front = P[order[int(0.55 * len(order)):]]  # closest ~45%
    c = front.mean(0)
    _, _, Vt = np.linalg.svd(front - c)
    nrm = Vt[-1]
    if nrm @ n_gt < 0:
        nrm = -nrm
    d = -nrm @ c
    return nrm / np.linalg.norm(nrm), float(d)


def segment(frames, gt: GroundTruth, alpha: float, ysign: int = 1,
            grid_margin: float = 0.05) -> Seg:
    neds, intens, fids = [], [], []
    for i, f in enumerate(frames):
        neds.append(apply_roll(axis_map(f.xyz.astype(float), ysign), alpha))
        intens.append(f.intensity.astype(float))
        fids.append(np.full(f.xyz.shape[0], i))
    ned = np.vstack(neds); inten = np.concatenate(intens); fid = np.concatenate(fids)

    n, e_u, e_v, O = plane_basis(gt)
    d = gt.plane_d
    R = np.linalg.norm(ned, axis=1)
    nP = ned @ n
    ahead = nP < -1e-9
    t = np.where(ahead, -d / np.where(ahead, nP, 1.0), 0.0)
    Q = ned * t[:, None]
    u = (Q - O) @ e_u
    v = (Q - O) @ e_v
    rects = window_rects_uv(gt, e_u, e_v, O)
    in_win = np.zeros(ned.shape[0], bool)
    for (u0, u1, v0, v1) in rects:
        in_win |= (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    gu0 = min(r[0] for r in rects); gu1 = max(r[1] for r in rects)
    gv0 = min(r[2] for r in rects); gv1 = max(r[3] for r in rects)
    m = grid_margin
    in_bbox = (u >= gu0 - m) & (u <= gu1 + m) & (v >= gv0 - m) & (v <= gv1 + m)
    cls = np.full(ned.shape[0], OFF)
    cls[ahead & in_bbox & ~in_win] = FRAME
    cls[ahead & in_win] = GLASS

    behind_gt = -(ned @ n + d)
    theta_gt = np.arccos(np.clip(np.abs(nP) / np.maximum(R, 1e-9), 0, 1))

    nf, df = _fit_front_plane(ned, cls, behind_gt, n)
    if df is None:
        nf, df = n, d
    nPf = ned @ nf
    behind_fit = -(ned @ nf + df)
    theta_fit = np.arccos(np.clip(np.abs(nPf) / np.maximum(R, 1e-9), 0, 1))

    return Seg(ned=ned, intensity=inten, cls=cls, R=R, theta_gt=theta_gt,
               behind_gt=behind_gt, theta_fit=theta_fit, behind_fit=behind_fit,
               u=u, v=v, fit_normal=nf, fit_d=df, frame_id=fid)


def load_segment(bag: str, gt_path: str, alpha: float, ysign: int = 1) -> Seg:
    return segment(read_all(bag), load_gt(gt_path), alpha, ysign)
