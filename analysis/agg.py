"""Shared aggregator: load+segment a scan selection -> per-point arrays.

Returns glass and frame per-point arrays (incidence angle, intensity, range,
depth-behind GT plane and fitted front) plus per-scan registration info. Frame
subsampling (stride) keeps it fast; pose is static within a bag.
"""
from __future__ import annotations

import os
import numpy as np

import sys
sys.path.insert(0, ".")
from analysis.seg import segment, GLASS, FRAME           # noqa: E402
from calibration.io_bag import read_all                   # noqa: E402
from calibration.gt_parse import load_gt                  # noqa: E402

CFG = {
    "oxy":  ("Oxy_L2_All/Oxy_L2_test_{n}",
             "Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_{n}.txt", list(range(1, 60)),
             {"near": list(range(31, 60)), "far": list(range(1, 31)),
              "frontal": [1, 11, 20, 31, 43]}),
    "bech": ("Bech_all/Bech_109_test_{n}",
             "Bech_Text_All/Bech_Text_All/Bech_109_test_{n}.txt", list(range(1, 41)),
             {"near": list(range(1, 26)), "far": list(range(26, 41)),
              "frontal": [1, 26]}),
    "l6":   ("L6_all/L6_all/L6_test_{n}",
             "L6_tests_1/L6_tests_1/L6_test_{n}.txt", list(range(1, 44)),
             {"frontal": [1, 9, 20]}),
}


def collect(ds, alpha=24.0, ysign=1, which="all", stride=8, reg_max=5.0):
    """Aggregate. Scans whose fitted-plane disagrees with GT by > reg_max deg are
    dropped (bad plane fit / extreme geometry) so depth/angle stay trustworthy."""
    bagt, gtt, allns, subsets = CFG[ds]
    ns = subsets.get(which, allns)
    G = {k: [] for k in ("th", "I", "R", "be", "bf")}
    F = {k: [] for k in ("th", "I", "R", "be", "bf")}
    reg = []
    used = dropped = 0
    for n in ns:
        bag, gtp = bagt.format(n=n), gtt.format(n=n)
        if not os.path.isdir(bag):
            continue
        frames = read_all(bag)
        if stride > 1:
            frames = frames[::stride]
        gt = load_gt(gtp)
        s = segment(frames, gt, alpha, ysign)
        cosang = abs(s.fit_normal @ gt.plane_normal /
                     (np.linalg.norm(s.fit_normal) * np.linalg.norm(gt.plane_normal)))
        ang = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
        reg.append((n, ang))
        if ang > reg_max:
            dropped += 1
            continue
        used += 1
        for d, m in ((G, s.cls == GLASS), (F, s.cls == FRAME)):
            d["th"].append(np.degrees(s.theta_fit[m]))
            d["I"].append(s.intensity[m]); d["R"].append(s.R[m])
            d["be"].append(s.behind_gt[m]); d["bf"].append(s.behind_fit[m])
    G = {k: np.concatenate(v) if v else np.array([]) for k, v in G.items()}
    F = {k: np.concatenate(v) if v else np.array([]) for k, v in F.items()}
    print(f"{ds}/{which}: used {used} scans (dropped {dropped} for reg>{reg_max}deg), "
          f"glass={G['I'].size}, frame={F['I'].size}")
    return G, F, reg


def binned_median(x, y, edges, minn=25):
    ctr, med, q1, q3, cnt = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= minn:
            ctr.append(0.5 * (a + b)); med.append(np.median(y[m]))
            q1.append(np.percentile(y[m], 25)); q3.append(np.percentile(y[m], 75))
            cnt.append(int(m.sum()))
    return (np.array(ctr), np.array(med), np.array(q1), np.array(q3), np.array(cnt))
