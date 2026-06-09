"""Fit the dataset-global correction and show it on one scan (frame on columns? glass in panes?)."""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, ".")
from calibration.io_bag import read_all
from calibration.gt_parse import load_gt
from calibration.label import detect_floor
from calibration.frames import apply_roll
from analysis.seg import axis_map
from calibration import register as reg

CFG = {"oxy": ("Oxy_L2_All/Oxy_L2_test_{n}", "Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_{n}.txt", range(1, 60)),
       "bech": ("Bech_all/Bech_109_test_{n}", "Bech_Text_All/Bech_Text_All/Bech_109_test_{n}.txt", range(1, 41))}
COL = {0: "gold", 1: "deepskyblue", 2: "limegreen", 3: "magenta"}
NM = {0: "frame", 1: "glass", 2: "ground", 3: "interior"}


def agg(bag):
    fr = read_all(bag)
    ned = np.vstack([apply_roll(axis_map(f.xyz.astype(float), 1), 24.0) for f in fr])
    inten = np.concatenate([f.intensity.astype(float) for f in fr])
    return fr, ned, inten


def main():
    ds, shown = sys.argv[1], int(sys.argv[2])
    bagt, gtt, ns = CFG[ds]
    present = [n for n in ns if os.path.isdir(bagt.format(n=n))]
    fit_ns = present[:: max(1, len(present) // 8)][:8]
    items = [(agg(bagt.format(n=n))[1], agg(bagt.format(n=n))[2], load_gt(gtt.format(n=n))) for n in fit_ns]
    corr = reg.register_dataset(items)
    print(f"{ds} GLOBAL: mp={corr.mp:+.3f} dd={corr.dd:+.3f} du={corr.du:+.3f} "
          f"dv={corr.dv:+.3f} da={corr.da:+.1f} trim={corr.delta:.3f}")

    fr, ned, inten = agg(bagt.format(n=shown))
    gt = load_gt(gtt.format(n=shown))
    lab = reg.label(ned, inten, gt, corr, floor=detect_floor(ned, gt))
    for k in (0, 1, 2, 3):
        print(f"  {NM[k]}: {100*(lab==k).mean():.1f}%")
    u, v, behind, R, ahead = reg.project(ned, gt)
    uc, vc = reg._apply_uv(u, v, corr)
    rects, (gu0, gu1, gv0, gv1) = reg._rects_bbox(gt)
    near = ahead & ((np.abs(behind - corr.mp) < 0.09) | (np.abs(behind - corr.dd) < 0.09))
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for k in (1, 0):
        m = (lab == k) & near
        if m.any():
            ax[0].scatter(uc[m], -vc[m], s=4, c=COL[k], label=f"{NM[k]} ({m.sum()})", alpha=0.7)
    for (u0, u1, v0, v1) in rects:
        d = corr.delta
        ax[0].add_patch(Rectangle((u0+d, -(v1-d)), (u1-u0)-2*d, (v1-v0)-2*d, fill=False, ec="red", lw=1.3))
    ax[0].set_xlim(gu0-0.5, gu1+0.5); ax[0].set_ylim(-(gv1+0.5), -(gv0-0.5)); ax[0].set_aspect("equal")
    ax[0].legend(); ax[0].set_title(f"{ds} test {shown} NEW labels (near-plane: frame+glass)")
    sc = ax[1].scatter(uc[near], -vc[near], s=3, c=inten[near], cmap="turbo", vmin=20, vmax=255)
    for (u0, u1, v0, v1) in rects:
        d = corr.delta
        ax[1].add_patch(Rectangle((u0+d, -(v1-d)), (u1-u0)-2*d, (v1-v0)-2*d, fill=False, ec="black", lw=1.0))
    ax[1].set_xlim(gu0-0.5, gu1+0.5); ax[1].set_ylim(-(gv1+0.5), -(gv0-0.5)); ax[1].set_aspect("equal")
    plt.colorbar(sc, ax=ax[1], shrink=0.7); ax[1].set_title("near-plane intensity (bright structure BETWEEN red rects?)")
    fig.tight_layout(); fig.savefig(f"analysis/global_{ds}_{shown}.png", dpi=110)
    print(f"wrote analysis/global_{ds}_{shown}.png")


if __name__ == "__main__":
    main()
