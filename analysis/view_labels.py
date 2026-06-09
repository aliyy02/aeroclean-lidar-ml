"""Plot a re-labeled scan in the de-rotated in-plane (uc,vc), colored by label, to check
whether GLASS (blue) still coincides with FRAME (yellow)."""
import sys
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, ".")
from calibration.gt_parse import load_gt
from calibration import register as reg

GT = {"oxy": "Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_{n}.txt",
      "bech": "Bech_Text_All/Bech_Text_All/Bech_109_test_{n}.txt"}
NAME = {0: "frame", 1: "glass", 2: "ground", 3: "interior"}
COL = {0: "gold", 1: "deepskyblue", 2: "limegreen", 3: "magenta"}


def main():
    ds, n = sys.argv[1], int(sys.argv[2])
    d = f"data/labeled_{ds}/{ds.capitalize()}_{n}"
    files = sorted(glob.glob(f"{d}/frame_*.npz"))[::4]
    xyz = np.vstack([np.load(f)["xyz"] for f in files])
    lab = np.concatenate([np.load(f)["label"] for f in files])
    corr_d = json.load(open(f"{d}/meta.json"))["correction"]
    corr = reg.Corr(**corr_d)
    gt = load_gt(GT[ds].format(n=n))
    u, v, behind, R, ahead = reg.project(xyz, gt)
    uc, vc = reg._apply_uv(u, v, corr)
    rects, (gu0, gu1, gv0, gv1) = reg._rects_bbox(gt)

    near = np.abs(behind - corr.dd) < 0.12          # frame + glass live here
    fig, ax = plt.subplots(figsize=(15, 8))
    for k in (1, 0):                                # glass then frame on top
        m = (lab == k) & near
        if m.any():
            ax.scatter(uc[m], -vc[m], s=5, c=COL[k], label=f"{NAME[k]} ({m.sum()})", alpha=0.6)
    for (u0, u1, v0, v1) in rects:
        dd = corr.delta
        ax.add_patch(Rectangle((u0 + dd, -(v1 - dd)), (u1 - u0) - 2 * dd, (v1 - v0) - 2 * dd,
                               fill=False, ec="red", lw=1.3))
    ax.set_xlim(gu0 - 0.5, gu1 + 0.5); ax.set_ylim(-(gv1 + 0.5), -(gv0 - 0.5))
    ax.set_aspect("equal"); ax.legend(markerscale=3)
    ax.set_title(f"{ds} test {n}: labels in de-rotated (u,v) -- does blue(glass) overlap gold(frame)?")
    fig.tight_layout(); fig.savefig(f"analysis/labels_{ds}_{n}.png", dpi=115)
    print(f"wrote analysis/labels_{ds}_{n}.png  glass={int((lab==1).sum())} frame={int((lab==0).sum())}")


if __name__ == "__main__":
    main()
