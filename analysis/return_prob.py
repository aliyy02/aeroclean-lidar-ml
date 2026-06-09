"""Relative return-probability vs incidence angle, range, material.

The bags hold ONLY returns (no missed-beam record: cloud is unorganized, is_dense,
ring=const), so absolute return rate isn't recoverable. But a FIXED forward beam-cone
in the (mount-fixed) sensor frame contains a fixed number of fired beams regardless of
pose; as the rig tilts, that cone is filled by the facade at the pose's incidence angle.
So (glass returns in the cone, per frame) is proportional to P_return(theta) * const.
Normalizing by total returns/frame removes any global rate drift. The two standoff
groups give the range dependence at matched angle.

Usage (ROS sourced):
  PYTHONPATH=.:$PYTHONPATH python3 analysis/return_prob.py [cone_deg] [stride]
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from analysis.seg import segment, GLASS, FRAME           # noqa: E402
from analysis.agg import CFG                              # noqa: E402
from calibration.io_bag import read_all                  # noqa: E402
from calibration.gt_parse import load_gt                 # noqa: E402


def scan_rows(ds, cone_deg, stride):
    bagt, gtt, allns, subsets = CFG[ds]
    near = set(subsets.get("near", []))
    rows = []
    cc = np.cos(np.radians(cone_deg))
    for n in allns:
        bag, gtp = bagt.format(n=n), gtt.format(n=n)
        if not os.path.isdir(bag):
            continue
        frames = read_all(bag)
        nf = len(frames)
        if stride > 1:
            frames = frames[::stride]; nf = len(frames)
        gt = load_gt(gtp)
        s = segment(frames, gt, 24.0, 1)
        R = np.maximum(s.R, 1e-9)
        fwd = (s.ned[:, 0] / R) > cc           # within cone of +x boresight
        gl = fwd & (s.cls == GLASS)
        if gl.sum() < 30:
            continue
        grp = "near" if n in near else "far"
        rows.append(dict(
            n=n, grp=grp,
            inc=float(np.degrees(np.median(s.theta_fit[gl]))),
            glass_per_frame=gl.sum() / nf,
            glass_per_total=gl.sum() / max((s.cls == GLASS).sum(), 1),
            total_per_frame=s.ned.shape[0] / nf,
            Rmed=float(np.median(R[gl])),
        ))
    return rows


def main():
    cone = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    print(f"forward cone = +/-{cone} deg about boresight\n")
    for j, ds in enumerate(("l6", "oxy", "bech")):
        rows = scan_rows(ds, cone, stride)
        inc = np.array([r["inc"] for r in rows])
        gpf = np.array([r["glass_per_frame"] for r in rows])
        grp = np.array([r["grp"] for r in rows])
        Rm = np.array([r["Rmed"] for r in rows])
        for g, col in (("near", "tab:blue"), ("far", "tab:red")):
            m = grp == g
            if m.any():
                ax[j].scatter(inc[m], gpf[m], c=col, label=f"{g} (R~{np.median(Rm[m]):.2f}m)",
                              s=40, alpha=0.8)
        ax[j].set_title(f"{ds}: glass returns/frame in forward cone vs incidence")
        ax[j].set_xlabel("incidence angle (deg)")
        ax[j].set_ylabel("glass returns / frame (in cone)")
        ax[j].legend(); ax[j].set_ylim(bottom=0)
        # range test at matched low incidence
        low = inc < 12
        for g in ("near", "far"):
            mm = low & (grp == g)
            if mm.any():
                print(f"  {ds} {g}: near-normal(<12deg) glass/frame in cone = "
                      f"{np.median(gpf[mm]):6.1f}  (R~{np.median(Rm[mm]):.2f}m, n={mm.sum()})")
        print()
    fig.tight_layout(); fig.savefig("analysis/return_prob.png", dpi=110)
    print("wrote analysis/return_prob.png")


if __name__ == "__main__":
    main()
