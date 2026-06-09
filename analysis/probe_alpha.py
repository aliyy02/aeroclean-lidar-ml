"""Sweep boresight roll alpha; report glass/frame separation (depth + intensity).

For each alpha we classify beams by GT aperture (glass=in-window, frame=in-grid),
aggregate over several scans, and report how distinctly the two classes separate
in depth-behind-plane and in intensity. The alpha that maximizes the (physical)
separation is the best mount-roll estimate AND evidence the materials separate.

Usage (ROS sourced):
  PYTHONPATH=.:$PYTHONPATH python3 analysis/probe_alpha.py oxy|bech [ysign] [alphas]
"""
import sys
import glob
import os
import numpy as np

sys.path.insert(0, ".")
from analysis.seg import segment, GLASS, FRAME            # noqa: E402
from calibration.io_bag import read_all                    # noqa: E402
from calibration.gt_parse import load_gt                   # noqa: E402

CFG = {
    "oxy":  ("Oxy_L2_All/Oxy_L2_test_{n}",
             "Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_{n}.txt",
             [1, 11, 20, 31, 43]),
    "bech": ("Bech_all/Bech_109_test_{n}",
             "Bech_Text_All/Bech_Text_All/Bech_109_test_{n}.txt",
             [1, 26]),
}


def main():
    ds = sys.argv[1]
    ysign = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    alphas = ([float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3
              else list(np.arange(8, 37, 2.0)))
    bagt, gtt, ns = CFG[ds]
    scans = [(bagt.format(n=n), gtt.format(n=n)) for n in ns
             if os.path.isdir(bagt.format(n=n))]
    # preload frames + gt once
    loaded = [(read_all(b), load_gt(g)) for b, g in scans]
    print(f"{ds}: {len(loaded)} scans, ysign={ysign}")
    print(f"{'alpha':>6} {'nG':>7} {'nF':>7} | {'behindG':>8} {'behindF':>8} "
          f"{'dDepth':>7} | {'intG':>6} {'intF':>6} {'dInt':>6}")
    best = None
    for al in alphas:
        bg, bf, ig, ifr = [], [], [], []
        for frames, gt in loaded:
            s = segment(frames, gt, al, ysign)
            bg.append(s.behind_gt[s.cls == GLASS])
            bf.append(s.behind_gt[s.cls == FRAME])
            ig.append(s.intensity[s.cls == GLASS])
            ifr.append(s.intensity[s.cls == FRAME])
        bg = np.concatenate(bg); bf = np.concatenate(bf)
        ig = np.concatenate(ig); ifr = np.concatenate(ifr)
        mbg, mbf = np.median(bg), np.median(bf)
        mig, mif = np.median(ig), np.median(ifr)
        ddep, dint = mbg - mbf, mig - mif
        score = abs(ddep)
        print(f"{al:6.1f} {bg.size:7d} {bf.size:7d} | {mbg:8.3f} {mbf:8.3f} "
              f"{ddep:7.3f} | {mig:6.1f} {mif:6.1f} {dint:6.1f}")
        if best is None or score > best[1]:
            best = (al, score)
    print(f"\nmax |depth-separation| at alpha={best[0]:.1f}")


if __name__ == "__main__":
    main()
