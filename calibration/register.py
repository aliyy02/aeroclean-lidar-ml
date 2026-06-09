"""Register the GT window grid to a real scan (undo a GLOBAL cm-scale tape error), then label.

User's scheme (authoritative): the GT corners lie on the glass panes.
  in-rectangle (y,z), within `tau_int` of the glass plane  -> GLASS
  in-rectangle, behind the glass plane by > tau_int         -> INTERIOR (see-through)
  outside the apertures (incl. proud wall columns / mullions, side+top columns) -> FRAME
  floor                                                     -> GROUND

The GT has ONE global error per panel (a couple cm, up to ~15 cm; lateral + depth). We fit:
  mp    : depth of the PROUD structure (mullions ~flush, or concrete columns proud of the glass)
  dd    : the GLASS plane depth (= GT plane +/- the small tape error)
  du,dv : in-plane shift,  da : small roll refinement,  delta : inward rect trim
by a search that puts the BRIGHT proud structure BETWEEN rects and the DIM glass INSIDE. The
search band spans mp..dd so both the bright columns and the (recessed) glass are present to
give the contrast. The error is global, so fit it ONCE per dataset (`register_dataset`).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from analysis.seg import plane_basis, window_rects_uv

NOT_GLASS, GLASS, GROUND, INTERIOR, DROP = 0, 1, 2, 3, -1


@dataclass
class Corr:
    dd: float = 0.0       # glass-plane depth (rel GT plane)
    du: float = 0.0
    dv: float = 0.0
    da: float = 0.0       # degrees
    delta: float = 0.0    # inward rect trim
    mp: float = 0.0       # proud-structure depth (columns/mullions); = dd when flush


def project(ned, gt):
    """Beam->GT-plane intersection in-plane (u,v), signed depth behind plane, range."""
    n, e_u, e_v, O = plane_basis(gt)
    d = gt.plane_d
    R = np.linalg.norm(ned, axis=1)
    nP = ned @ n
    ahead = nP < -1e-9
    t = np.where(ahead, -d / np.where(ahead, nP, 1.0), 0.0)
    Q = ned * t[:, None]
    u = (Q - O) @ e_u
    v = (Q - O) @ e_v
    behind = -(ned @ n + d)
    return u, v, behind, R, ahead


def _apply_uv(u, v, corr: Corr):
    """Move cloud (u,v) into the corrected GT frame: rotate by -da, shift by -(du,dv)."""
    a = np.radians(-corr.da)
    c, s = np.cos(a), np.sin(a)
    return c * u - s * v - corr.du, s * u + c * v - corr.dv


def _rects_bbox(gt):
    n, e_u, e_v, O = plane_basis(gt)
    rects = window_rects_uv(gt, e_u, e_v, O)
    gu0 = min(r[0] for r in rects); gu1 = max(r[1] for r in rects)
    gv0 = min(r[2] for r in rects); gv1 = max(r[3] for r in rects)
    return rects, (gu0, gu1, gv0, gv1)


def in_rect_mask(uc, vc, rects, delta):
    m = np.zeros(uc.shape[0], bool)
    for (u0, u1, v0, v1) in rects:
        m |= (uc >= u0 + delta) & (uc <= u1 - delta) & (vc >= v0 + delta) & (vc <= v1 - delta)
    return m


def _estimate_planes(behind, inten, in_grid, in_rect0):
    """mp = proud structure (bright in-grid front); dd = glass plane (in-rect front edge)."""
    front = in_grid & (behind > -0.30) & (behind < 0.60)
    if int(front.sum()) < 100:
        return 0.0, 0.0
    gthr = np.percentile(inten[front], 55)
    fb = front & (inten >= gthr)
    mp = float(np.clip(np.percentile(behind[fb if int(fb.sum()) > 100 else front], 10),
                       -0.25, 0.15))
    # glass plane = the FRONT edge of the in-rect returns (the pane surface). Use a LOW
    # percentile so a sparse glass front isn't outvoted by a dense curtain/see-through behind
    # it, and clamp to a small global tape error (corners are on the glass).
    inr_front = in_rect0 & (behind > -0.15) & (behind < 0.60)
    dd = (float(np.clip(np.percentile(behind[inr_front], 3), -0.07, 0.07))
          if int(inr_front.sum()) > 100 else mp)
    return mp, dd


def _search(un, vn, bn, rects, bbox, du_rng, dv_rng, step, da_rng, da_step, deltas,
            cap=15000, seed=0):
    """Find (du,dv,da,delta) by maximizing the DEPTH CONTRAST: returns inside the apertures
    are DEEP (curtain / see-through behind the glass) while the mullions/columns return at the
    FRONT. So `median(behind | in-rect) - median(behind | out-of-rect)` peaks at the true
    alignment. This signal is intrinsic to the panel (can't be gamed by trimming/shifting off,
    which collapses the contrast) and is strong for both Oxy (curtain) and Bech (see-through).
    `bn` = depth-behind-plane of each point."""
    if un.shape[0] < 80:
        return 0.0, 0.0, 0.0, 0.0
    if un.shape[0] > cap:
        idx = np.random.default_rng(seed).choice(un.shape[0], cap, replace=False)
        un, vn, bn = un[idx], vn[idx], bn[idx]
    gu0, gu1, gv0, gv1 = bbox
    dus = np.arange(du_rng[0], du_rng[1] + 1e-9, step)
    dvs = np.arange(dv_rng[0], dv_rng[1] + 1e-9, step)
    das = np.arange(da_rng[0], da_rng[1] + 1e-9, da_step)
    best, best_t = -1e9, (0.0, 0.0, 0.0, 0.0)
    for da in das:
        a = np.radians(-da); c, s = np.cos(a), np.sin(a)
        ur, vr = c * un - s * vn, s * un + c * vn
        for du in dus:
            uc = ur - du
            for dv in dvs:
                vc = vr - dv
                in_bbox = (uc >= gu0) & (uc <= gu1) & (vc >= gv0) & (vc <= gv1)
                if int(in_bbox.sum()) < 80:
                    continue
                for delta in deltas:
                    inr = in_rect_mask(uc, vc, rects, delta)
                    inn = in_bbox & inr
                    out = in_bbox & ~inr
                    if int(inn.sum()) < 30 or int(out.sum()) < 30:
                        continue
                    score = float(np.median(bn[inn]) - np.median(bn[out]))
                    if score > best:
                        best, best_t = score, (float(du), float(dv), float(da), float(delta))
    return best_t


def _near_band(behind, ahead, mp, dd, band):
    lo, hi = min(mp, dd) - band, max(mp, dd) + band
    return ahead & (behind >= lo) & (behind <= hi)


def register(ned, inten, gt, band=0.07, du_rng=(-0.12, 0.12), dv_rng=(-0.12, 0.12),
             step=0.015, da_rng=(-3, 3), da_step=1.0,
             deltas=(0.0, 0.02, 0.03, 0.04)) -> Corr:
    """Per-scan global correction."""
    u, v, behind, R, ahead = project(ned, gt)
    rects, bbox = _rects_bbox(gt)
    gu0, gu1, gv0, gv1 = bbox
    in_grid = ahead & (u > gu0 - 0.1) & (u < gu1 + 0.1) & (v > gv0 - 0.1) & (v < gv1 + 0.1)
    in_rect0 = ahead & in_rect_mask(u, v, rects, 0.0)
    mp, dd = _estimate_planes(behind, inten, in_grid, in_rect0)
    facade = ahead & in_grid & (behind > mp - 0.30) & (behind < dd + 1.5)
    du, dv, da, delta = _search(u[facade], v[facade], behind[facade], rects, bbox,
                                du_rng, dv_rng, step, da_rng, da_step, deltas)
    return Corr(dd=dd, mp=mp, du=du, dv=dv, da=da, delta=delta)


def register_dataset(items, band=0.07, du_rng=(-0.12, 0.12), dv_rng=(-0.12, 0.12),
                     step=0.015, da_rng=(-3, 3), da_step=1.0,
                     deltas=(0.0, 0.02, 0.03, 0.04), sub=20000, seed=0) -> Corr:
    """ONE global correction fit from many scans (the tape error is the same for all).
    items: iterable of (ned, inten, gt)."""
    rng = np.random.default_rng(seed)
    rects = bbox = None
    cu, cv, cb, ci, cg, cr = [], [], [], [], [], []
    for ned, inten, gt in items:
        u, v, behind, R, ahead = project(ned, gt)
        if rects is None:
            rects, bbox = _rects_bbox(gt)
        gu0, gu1, gv0, gv1 = bbox
        ig = ahead & (u > gu0 - 0.1) & (u < gu1 + 0.1) & (v > gv0 - 0.1) & (v < gv1 + 0.1)
        ir = ahead & in_rect_mask(u, v, rects, 0.0)
        idx = np.where(ahead)[0]
        if idx.size > sub:
            idx = rng.choice(idx, sub, replace=False)
        cu.append(u[idx]); cv.append(v[idx]); cb.append(behind[idx])
        ci.append(inten[idx]); cg.append(ig[idx]); cr.append(ir[idx])
    u = np.concatenate(cu); v = np.concatenate(cv); behind = np.concatenate(cb)
    inten = np.concatenate(ci); in_grid = np.concatenate(cg); in_rect0 = np.concatenate(cr)
    mp, dd = _estimate_planes(behind, inten, in_grid, in_rect0)
    facade = in_grid & (behind > mp - 0.30) & (behind < dd + 1.5)
    du, dv, da, delta = _search(u[facade], v[facade], behind[facade], rects, bbox,
                                du_rng, dv_rng, step, da_rng, da_step, deltas)
    return Corr(dd=dd, mp=mp, du=du, dv=dv, da=da, delta=delta)


def reclassify_glass_at_frame(lab, uc, vc, behind=None, dd=0.0, eps=0.03, iters=12,
                              interior_back=0.35, cap=40000, seed=0):
    """User rule: structure sharing ~the same de-rotated in-plane (uc,vc) as a FRAME point is
    really FRAME -- the mullions/columns (and recessed inner frames) protrude past / sit behind
    the thin GT rectangle edge, so in-aperture points on them were mislabeled glass or interior.
    Because that structure is CONNECTED to the outer frame, this REGION-GROWS: convert points
    within `eps` of frame, fold into frame, repeat. It targets GLASS always, and -- when `behind`
    is given -- also the NEAR-PLANE interior (within dd+interior_back: the recessed inner frames),
    while leaving the deep see-through / open voids alone. Run on the AGGREGATE scan (all frames
    stacked) so the footprint is dense. uc,vc are the de-rotated coords (after the roll)."""
    from scipy.spatial import cKDTree
    lab = np.asarray(lab).copy()
    pts = np.column_stack([np.asarray(uc, float), np.asarray(vc, float)])
    rng = np.random.default_rng(seed)
    near = (np.asarray(behind, float) < dd + interior_back) if behind is not None else None
    for _ in range(iters):
        cand = (lab == GLASS)
        if near is not None:
            cand = cand | ((lab == INTERIOR) & near)
        g = np.where(cand)[0]
        f = np.where(lab == NOT_GLASS)[0]
        if g.size == 0 or f.size == 0:
            break
        fpts = pts[f] if f.size <= cap else pts[rng.choice(f, cap, replace=False)]
        d, _ = cKDTree(fpts).query(pts[g], k=1)
        conv = g[d < eps]
        if conv.size == 0:
            break
        lab[conv] = NOT_GLASS
    return lab


def label(ned, inten, gt, corr: Corr, floor=None, tau_front=0.025, tau_int=0.07,
          frame_front=0.50, frame_back=0.40, region_margin=0.45):
    """User's scheme. Classify by each point's PERPENDICULAR position on the fitted glass plane
    (where the point actually sits), NOT where its beam crosses the plane -- so a frame sitting
    BEHIND the glass (the beam goes through the pane first) still reads as frame at its mullion
    y-z instead of being mislabeled interior. glass = in-window within tau_int (~glass thickness)
    of the plane; interior = in-window behind that (see-through to the room); frame = OUTSIDE the
    apertures, from the proud columns (frame_front in front) to recessed inner frames (frame_back
    behind); ground = floor."""
    n, e_u, e_v, O = plane_basis(gt)
    rel = ned - O
    u = rel @ e_u
    v = rel @ e_v                                    # perpendicular (actual) in-plane position
    behind = -(ned @ n + gt.plane_d)
    ahead = (ned @ n) < -1e-9
    uc, vc = _apply_uv(u, v, corr)
    bc = behind - corr.dd
    rects, (gu0, gu1, gv0, gv1) = _rects_bbox(gt)
    inr = in_rect_mask(uc, vc, rects, corr.delta)
    m = region_margin
    in_region = ahead & (uc >= gu0 - m) & (uc <= gu1 + m) & (vc >= gv0 - m) & (vc <= gv1 + m)
    facade_depth = (behind > corr.dd - frame_front) & (behind < corr.dd + frame_back)

    on_floor = np.zeros(ned.shape[0], bool)
    if floor is not None:
        fn = np.asarray(floor[0], float); fd = float(floor[1])
        n, e_u, e_v, O = plane_basis(gt)
        fu = (ned - O) @ e_u
        on_floor = (np.abs(ned @ fn + fd) <= 0.05) & (fu >= gu0 - 0.3) & (fu <= gu1 + 0.3)

    free = ~on_floor
    lab = np.full(ned.shape[0], DROP, int)
    lab[in_region & ~inr & facade_depth & free] = NOT_GLASS
    lab[ahead & inr & (bc >= -tau_front) & (bc <= tau_int) & free] = GLASS
    lab[ahead & inr & (bc > tau_int) & free] = INTERIOR
    lab[on_floor] = GROUND
    return lab
