"""TDD for the GT->data registration + user's labeling scheme (offline, synthetic).

Bug reproduced: the per-scan search clamps at its range boundaries, so a real tape error
larger than the (old) +/-0.07 window can't be recovered -> the grid stays misaligned ->
'almost no glass'. These tests pin: (1) register recovers a shift beyond the old range,
(2) a dataset-global fit recovers it from several scans, (3) label() implements the user's
scheme (in-rect within 8cm of the glass plane = glass; behind = interior; outside the
apertures = frame; floor = ground).
"""
import copy
import numpy as np

from calibration.gt_parse import parse_gt
from calibration import register as reg

GT_TEXT = """6-DOF Test Bed
Pose: X=0 Y=0 Z=0 | Yaw=0 Pitch=0 Roll=0
LiDAR (bed frame): [0,0,0]
  r1c1:
    UL: [2000.0, -600.0, -400.0]
    UR: [2000.0, -100.0, -400.0]
    LL: [2000.0, -600.0, 400.0]
    LR: [2000.0, -100.0, 400.0]
  r1c2:
    UL: [2000.0, 100.0, -400.0]
    UR: [2000.0, 600.0, -400.0]
    LL: [2000.0, 100.0, 400.0]
    LR: [2000.0, 600.0, 400.0]
Facade plane: -1.0000x + 0.0000y + 0.0000z + 2000.0000 = 0
"""


def synth(seed=0):
    """Synthetic facade cloud (corrected NED): proud bright columns OUT of the apertures,
    dim glass IN the apertures at the plane (x=2.0), see-through behind, floor below."""
    gt = parse_gt(GT_TEXT)
    rng = np.random.default_rng(seed)
    P, I, T = [], [], []

    def add(xs, ys, zs, inten, lab):
        for x, y, z in zip(xs, ys, zs):
            P.append([x, y, z]); I.append(inten); T.append(lab)

    for (y0, y1) in [(-0.58, -0.12), (0.12, 0.58)]:                 # glass surface, dim, at plane
        add(np.full(200, 2.0), rng.uniform(y0, y1, 200), rng.uniform(-0.38, 0.38, 200),
            40.0, reg.GLASS)
    for (y0, y1) in [(-0.58, -0.12), (0.12, 0.58)]:                 # see-through, behind (0.40)
        add(np.full(140, 2.40), rng.uniform(y0, y1, 140), rng.uniform(-0.38, 0.38, 140),
            60.0, reg.INTERIOR)
    for (y0, y1) in [(-0.08, 0.08), (-0.95, -0.62), (0.62, 0.95)]:  # columns, bright, proud (-0.18)
        add(np.full(160, 1.82), rng.uniform(y0, y1, 160), rng.uniform(-0.40, 0.40, 160),
            220.0, reg.NOT_GLASS)
    for (y0, y1) in [(-0.08, 0.08), (-0.95, -0.62), (0.62, 0.95)]:  # frontmost column face (-0.40)
        add(np.full(80, 1.60), rng.uniform(y0, y1, 80), rng.uniform(-0.40, 0.40, 80),
            220.0, reg.NOT_GLASS)
    # recessed frame BEHIND the glass at an off-axis mullion y-z: its beam crosses the plane
    # *inside* a window (so beam->plane mislabels it interior), but it is frame.
    for (y0, y1) in [(0.61, 0.70), (-0.70, -0.61)]:
        add(np.full(60, 2.30), rng.uniform(y0, y1, 60), rng.uniform(-0.35, 0.35, 60),
            200.0, reg.NOT_GLASS)
    add(rng.uniform(0.6, 2.6, 260), rng.uniform(-0.7, 0.7, 260), np.full(260, 1.0),  # floor
        120.0, reg.GROUND)
    return gt, np.array(P, float), np.array(I, float), np.array(T, int)


def shift_gt_u(gt, s):
    """Shift the GT grid by +s along the in-plane horizontal (a lateral tape error)."""
    g = copy.deepcopy(gt)
    for w in g.windows.values():
        for k in w:
            w[k] = w[k] + np.array([0.0, s, 0.0])
    return g


def test_label_user_scheme():
    gt, ned, inten, truth = synth()
    floor = (np.array([0.0, 0.0, -1.0]), 1.0)            # plane z=1 (below), normal toward sensor
    lab = reg.label(ned, inten, gt, reg.Corr(dd=0.0, mp=-0.18), floor=floor, tau_int=0.08)
    for cls, name in [(reg.GLASS, "glass"), (reg.NOT_GLASS, "frame"),
                      (reg.INTERIOR, "interior"), (reg.GROUND, "ground")]:
        m = truth == cls
        acc = float((lab[m] == cls).mean())
        assert acc > 0.9, f"{name}: only {acc:.2f} labeled correctly"


def test_register_recovers_shift_beyond_old_range():
    gt, ned, inten, _ = synth()
    s = 0.11                                              # > old +/-0.07 search half-window
    corr = reg.register(ned, inten, shift_gt_u(gt, s))    # DEFAULT ranges
    assert abs(corr.du - (-s)) < 0.03, f"du={corr.du:+.3f}, expected ~{-s:+.3f}"


def test_glass_plane_not_biased_by_dense_curtain():
    """Oxy-like: a SPARSE glass front + a DENSE curtain behind, in the apertures. The
    estimated glass plane must sit at the front glass (~0), not at the dense curtain."""
    gt = parse_gt(GT_TEXT)
    rng = np.random.default_rng(2)
    P, I = [], []
    for (y0, y1) in [(-0.58, -0.12), (0.12, 0.58)]:
        for y, z in zip(rng.uniform(y0, y1, 15), rng.uniform(-0.38, 0.38, 15)):
            P.append([2.0, y, z]); I.append(40.0)          # sparse glass front at the plane
        for y, z in zip(rng.uniform(y0, y1, 300), rng.uniform(-0.38, 0.38, 300)):
            P.append([2.26, y, z]); I.append(130.0)        # dense curtain 0.26 behind
    ned, inten = np.array(P, float), np.array(I, float)
    u, v, behind, R, ahead = reg.project(ned, gt)
    rects, (gu0, gu1, gv0, gv1) = reg._rects_bbox(gt)
    in_grid = ahead & (u > gu0 - 0.1) & (u < gu1 + 0.1) & (v > gv0 - 0.1) & (v < gv1 + 0.1)
    in_rect0 = ahead & reg.in_rect_mask(u, v, rects, 0.0)
    mp, dd = reg._estimate_planes(behind, inten, in_grid, in_rect0)
    assert abs(dd) < 0.06, f"glass plane dd={dd:.3f} biased toward the dense curtain"


def test_glass_coinciding_with_frame_becomes_frame():
    """User rule: a 'glass' point sharing ~the same (de-rotated) y-z as a 'frame' point is
    really frame (the mullion/column structure protrudes past the thin GT rectangle edge)."""
    lab = np.array([reg.NOT_GLASS, reg.NOT_GLASS, reg.GLASS, reg.GLASS, reg.INTERIOR])
    uc = np.array([0.00, 0.50, 0.015, 2.00, 0.00])
    vc = np.array([0.00, 0.50, 0.000, 2.00, 0.00])
    out = reg.reclassify_glass_at_frame(lab, uc, vc, eps=0.03)
    assert out[2] == reg.NOT_GLASS, "glass next to a frame point -> frame"
    assert out[3] == reg.GLASS, "glass far from any frame -> stays glass"
    assert out[4] == reg.INTERIOR and out[0] == reg.NOT_GLASS, "other classes untouched"


def test_recessed_frame_interior_region_grows_to_frame():
    """A recessed frame structure lands in-window (interior) but is connected to the frame and
    sits near the plane -> frame. Deep see-through interior (far behind) stays interior."""
    lab = np.array([reg.NOT_GLASS, reg.INTERIOR, reg.INTERIOR])
    uc = np.array([0.00, 0.02, 0.02])
    vc = np.array([0.00, 0.00, 0.00])
    behind = np.array([0.00, 0.20, 1.50])
    out = reg.reclassify_glass_at_frame(lab, uc, vc, behind=behind, dd=0.0, eps=0.03,
                                        interior_back=0.35)
    assert out[1] == reg.NOT_GLASS, "near-plane interior connected to frame -> frame"
    assert out[2] == reg.INTERIOR, "deep see-through interior stays interior"


def test_register_dataset_recovers_global_shift():
    gt, ned, inten, _ = synth(1)
    s = 0.10
    gshift = shift_gt_u(gt, s)
    items = [(ned, inten, gshift) for _ in range(3)]      # same global error across 'scans'
    corr = reg.register_dataset(items)
    assert abs(corr.du - (-s)) < 0.03, f"du={corr.du:+.3f}, expected ~{-s:+.3f}"
