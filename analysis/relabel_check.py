"""Fit the global GT correction on a scan and show the re-labeling vs the old one.

Usage (ROS sourced):
  PYTHONPATH=.:$PYTHONPATH python3 analysis/relabel_check.py <ds> <N>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, ".")
from calibration.io_bag import read_all
from calibration.gt_parse import load_gt
from calibration.label import detect_floor
from analysis.seg import segment, plane_basis, window_rects_uv
from calibration import register as reg

CFG = {"oxy": ("Oxy_L2_All/Oxy_L2_test_{n}", "Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_{n}.txt"),
       "bech": ("Bech_all/Bech_109_test_{n}", "Bech_Text_All/Bech_Text_All/Bech_109_test_{n}.txt"),
       "l6": ("L6_all/L6_all/L6_test_{n}", "L6_tests_1/L6_tests_1/L6_test_{n}.txt")}
NAME = {0: "frame", 1: "glass", 2: "ground", 3: "interior", -1: "drop"}
COL = {0: "gold", 1: "deepskyblue", 2: "limegreen", 3: "magenta", -1: "lightgray"}


def main():
    ds, n = sys.argv[1], int(sys.argv[2])
    bag, gtp = CFG[ds][0].format(n=n), CFG[ds][1].format(n=n)
    frames = read_all(bag)[::3]
    gt = load_gt(gtp)
    s = segment(frames, gt, 24.0, 1)
    ned, inten = s.ned, s.intensity

    corr = reg.register(ned, inten, gt)
    floor = detect_floor(ned, gt)
    lab = reg.label(ned, inten, gt, corr, floor=floor)
    print(f"\n=== {ds} {n}  pose={gt.pose} ===")
    print(f"correction: dd={corr.dd:+.3f}m  du={corr.du:+.3f}  dv={corr.dv:+.3f}  "
          f"da={corr.da:+.1f}deg  delta(trim)={corr.delta:.3f}m")
    for k in (0, 1, 2, 3):
        m = lab == k
        print(f"  {NAME[k]:9s}: {m.sum():7d} ({100*m.mean():4.1f}%)")

    # view in corrected GT frame
    u, v, behind, R, ahead = reg.project(ned, gt)
    uc, vc = reg._apply_uv(u, v, corr)
    rects, _ = reg._rects_bbox(gt)
    rects0, (gu0, gu1, gv0, gv1) = reg._rects_bbox(gt)
    nearpl = ahead & ((np.abs(behind - corr.mp) < 0.09) |    # proud columns/mullions (frame)
                      (np.abs(behind - corr.dd) < 0.09))     # recessed glass plane (glass)
    fig, ax = plt.subplots(1, 2, figsize=(17, 8))
    for k in (1, 0):                          # near-plane only: frame (mullions) + glass (front)
        m = (lab == k) & nearpl
        if m.any():
            ax[0].scatter(uc[m], -vc[m], s=4, c=COL[k], label=f"{NAME[k]} ({m.sum()})", alpha=0.7)
    for (u0, u1, v0, v1) in rects:
        d = corr.delta
        ax[0].add_patch(Rectangle((u0 + d, -(v1 - d)), (u1 - u0) - 2 * d, (v1 - v0) - 2 * d,
                                  fill=False, ec="red", lw=1.2))
    ax[0].set_xlim(gu0 - 0.3, gu1 + 0.3); ax[0].set_ylim(-(gv1 + 0.3), -(gv0 - 0.3))
    ax[0].set_aspect("equal"); ax[0].legend(markerscale=4, fontsize=8)
    ax[0].set_title(f"NEW labels (corrected grid, red)\ndd={corr.dd:+.2f} du={corr.du:+.2f} "
                    f"dv={corr.dv:+.2f} da={corr.da:+.0f} trim={corr.delta:.2f}")
    fac = ahead & (np.abs(behind - corr.mp) < 0.07)
    sc = ax[1].scatter(uc[fac], -vc[fac], s=3, c=inten[fac], cmap="turbo", vmin=20, vmax=255)
    for (u0, u1, v0, v1) in rects:
        d = corr.delta
        ax[1].add_patch(Rectangle((u0 + d, -(v1 - d)), (u1 - u0) - 2 * d, (v1 - v0) - 2 * d,
                                  fill=False, ec="black", lw=1.0))
    ax[1].set_aspect("equal"); plt.colorbar(sc, ax=ax[1], shrink=0.7)
    ax[1].set_title("near-plane intensity (bright mullions should fall BETWEEN red rects)")
    fig.suptitle(f"{ds} test {n}")
    fig.tight_layout(); out = f"analysis/relabel_{ds}_{n}.png"
    fig.savefig(out, dpi=110); print(f"wrote {out}")


if __name__ == "__main__":
    main()
