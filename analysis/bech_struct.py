"""Examine Bech's real near-plane structure (perpendicular projection) vs the GT grid and the
current labels -- is the bright SOLID frame grid being labeled interior?"""
import sys
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, ".")
from calibration.gt_parse import load_gt
from analysis.seg import plane_basis
from calibration import register as reg

NAME = {0: "frame", 1: "glass", 2: "ground", 3: "interior"}
COL = {0: "gold", 1: "deepskyblue", 2: "limegreen", 3: "magenta"}


def main():
    ds, n = sys.argv[1], int(sys.argv[2])
    d = f"data/labeled_{ds}/{ds.capitalize()}_{n}"
    files = sorted(glob.glob(f"{d}/frame_*.npz"))[::3]
    xyz = np.vstack([np.load(f)["xyz"] for f in files])
    inten = np.concatenate([np.load(f)["intensity"] for f in files])
    lab = np.concatenate([np.load(f)["label"] for f in files])
    corr = reg.Corr(**json.load(open(f"{d}/meta.json"))["correction"])
    gt = load_gt(f"{'Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_' if ds=='oxy' else 'Bech_Text_All/Bech_Text_All/Bech_109_test_'}{n}.txt")
    nrm, e_u, e_v, O = plane_basis(gt)
    rel = xyz - O
    u = rel @ e_u; v = rel @ e_v
    behind = -(xyz @ nrm + gt.plane_d)
    uc, vc = reg._apply_uv(u, v, corr)
    rects, (gu0, gu1, gv0, gv1) = reg._rects_bbox(gt)

    near = (behind > corr.dd - 0.15) & (behind < corr.dd + 0.35)   # near + slightly recessed
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    s1 = ax[0].scatter(uc[near], -vc[near], s=4, c=inten[near], cmap="turbo", vmin=20, vmax=255)
    ax[0].set_title("near-plane INTENSITY (bright = solid frame grid)")
    plt.colorbar(s1, ax=ax[0], shrink=0.7)
    for k in (3, 1, 0):
        m = (lab == k) & near
        if m.any():
            ax[1].scatter(uc[m], -vc[m], s=4, c=COL[k], label=f"{NAME[k]} ({m.sum()})", alpha=0.6)
    ax[1].set_title("current LABELS near-plane (is the bright grid = interior?)")
    ax[1].legend()
    for a in ax:
        for (u0, u1, v0, v1) in rects:
            dd = corr.delta
            a.add_patch(Rectangle((u0 + dd, -(v1 - dd)), (u1 - u0) - 2 * dd, (v1 - v0) - 2 * dd,
                                  fill=False, ec="black", lw=1.3))
        a.set_xlim(gu0 - 0.4, gu1 + 0.4); a.set_ylim(-(gv1 + 0.4), -(gv0 - 0.4)); a.set_aspect("equal")
    fig.suptitle(f"{ds} test {n}: real structure vs GT grid (perpendicular projection)")
    fig.tight_layout(); fig.savefig(f"analysis/struct_{ds}_{n}.png", dpi=115)
    print(f"wrote analysis/struct_{ds}_{n}.png")
    # depth histogram of in-window vs out-window (perpendicular)
    inr = reg.in_rect_mask(uc, vc, rects, corr.delta)
    print(f"  in-window behind pct[5,25,50,75,95]={np.round(np.percentile(behind[inr],[5,25,50,75,95]),3)}")
    print(f"  out-window behind pct[5,25,50,75,95]={np.round(np.percentile(behind[~inr],[5,25,50,75,95]),3)}")
    # intensity of interior near plane (is it bright=frame?)
    intr_near = (lab == 3) & (behind < corr.dd + 0.15)
    print(f"  INTERIOR within 15cm of plane: {intr_near.sum()} pts, intensity med={np.median(inten[intr_near]) if intr_near.any() else 0:.0f}")


if __name__ == "__main__":
    main()
