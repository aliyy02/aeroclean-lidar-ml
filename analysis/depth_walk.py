"""Measure the depth structure the user described: walk in depth from the frontmost frame
to the glass plane, per dataset. Confirms the offsets (Bech ~30cm, Oxy ~11.7cm + a 2nd frame
~2.1cm before glass), the glass thickness (~5-6cm), and where 'interior' begins.

Separates OUT-of-window (frame/columns) from IN-window (glass+interior) using the fitted
global GT correction, then profiles depth-behind-GT-plane for each.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from calibration.io_bag import read_all
from calibration.gt_parse import load_gt
from calibration.frames import apply_roll
from analysis.seg import axis_map
from calibration import register as reg

CFG = {"oxy": ("Oxy_L2_All/Oxy_L2_test_{n}", "Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_{n}.txt",
               [1, 11, 20, 31, 43]),
       "bech": ("Bech_all/Bech_109_test_{n}", "Bech_Text_All/Bech_Text_All/Bech_109_test_{n}.txt",
                [1, 26])}


def agg(bag):
    fr = read_all(bag)
    ned = np.vstack([apply_roll(axis_map(f.xyz.astype(float), 1), 24.0) for f in fr])
    inten = np.concatenate([f.intensity.astype(float) for f in fr])
    return ned, inten


def main():
    ds = sys.argv[1]
    bagt, gtt, frontal = CFG[ds]
    # global correction (fit on a spread)
    allns = [n for n in range(1, 60) if os.path.isdir(bagt.format(n=n))]
    fit_ns = allns[:: max(1, len(allns) // 12)][:12]
    items = [(*agg(bagt.format(n=n)), load_gt(gtt.format(n=n))) for n in fit_ns]
    corr = reg.register_dataset(items)
    print(f"{ds} corr: mp={corr.mp:+.3f} dd={corr.dd:+.3f} du={corr.du:+.3f} dv={corr.dv:+.3f} da={corr.da:+.1f}")

    OUTb, INb, OUTi, INi = [], [], [], []
    for n in frontal:
        if not os.path.isdir(bagt.format(n=n)):
            continue
        ned, inten = agg(bagt.format(n=n))
        gt = load_gt(gtt.format(n=n))
        u, v, behind, R, ahead = reg.project(ned, gt)
        uc, vc = reg._apply_uv(u, v, corr)
        rects, (gu0, gu1, gv0, gv1) = reg._rects_bbox(gt)
        inr = reg.in_rect_mask(uc, vc, rects, corr.delta)
        m = 0.45
        in_region = ahead & (uc >= gu0 - m) & (uc <= gu1 + m) & (vc >= gv0 - m) & (vc <= gv1 + m)
        fac = in_region & (behind > corr.mp - 0.30) & (behind < corr.dd + 1.2)
        OUTb.append(behind[fac & ~inr]); INb.append(behind[fac & inr])
        OUTi.append(inten[fac & ~inr]); INi.append(inten[fac & inr])
    OUTb = np.concatenate(OUTb); INb = np.concatenate(INb)

    front = np.percentile(OUTb, 2)          # frontmost frame (robust)
    print(f"  frontmost frame (2pct out-window behind) = {front:+.3f} m (rel GT plane)")
    print(f"  out-window(frame) behind pct[2,25,50,75,98]= {np.round(np.percentile(OUTb,[2,25,50,75,98]),3)}")
    print(f"  in-window        behind pct[2,25,50,75,98]= {np.round(np.percentile(INb,[2,25,50,75,98]),3)}")
    # in-window front edge (glass plane) = low percentile of in-window behind
    glass = np.percentile(INb, 8)
    print(f"  in-window front edge (8pct) = glass plane ~ {glass:+.3f}  => offset from frontmost frame = {glass-front:+.3f} m")

    fig, ax = plt.subplots(figsize=(11, 5))
    bins = np.linspace(front - 0.05, corr.dd + 0.8, 160)
    ax.hist(OUTb, bins=bins, color="gold", alpha=0.6, label="OUT-of-window (frame/columns)", density=True)
    ax.hist(INb, bins=bins, color="dodgerblue", alpha=0.6, label="IN-window (glass+interior)", density=True)
    ax.axvline(front, color="k", ls="--", lw=1, label=f"frontmost frame {front:+.2f}")
    ax.axvline(glass, color="green", ls="--", lw=1.5, label=f"glass plane {glass:+.2f} (offset {glass-front:+.2f})")
    ax.axvline(glass + 0.06, color="red", ls=":", lw=1.5, label="glass+6cm (interior begins)")
    ax.set_xlabel("depth behind GT plane (m)  [more negative = closer to lidar]")
    ax.set_ylabel("density"); ax.set_title(f"{ds}: depth structure (frontal scans)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"analysis/depthwalk_{ds}.png", dpi=120)
    print(f"  wrote analysis/depthwalk_{ds}.png")


if __name__ == "__main__":
    main()
