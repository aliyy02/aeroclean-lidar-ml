"""Measure, from the CLEAN labels, the quantities the forward model must reproduce:
per class (frame/glass/interior/ground): intensity (median + saturation), and for INTERIOR the
depth-behind-plane distribution (the see-through model the current forward model lacks)."""
import sys
import glob
import json
import numpy as np

sys.path.insert(0, ".")
from calibration.gt_parse import load_gt
from analysis.seg import plane_basis
from calibration import register as reg

CFG = {"l6": ("data/labeled", "L6_test_{n}", "L6_tests_1/L6_tests_1/L6_test_{n}.txt",
              [1, 9, 20, 30, 43]),
       "oxy": ("data/labeled_oxy", "Oxy_{n}", "Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_{n}.txt",
               [1, 11, 20, 31, 43]),
       "bech": ("data/labeled_bech", "Bech_{n}", "Bech_Text_All/Bech_Text_All/Bech_109_test_{n}.txt",
                [1, 14, 21, 26, 34])}
NAME = {0: "frame", 1: "glass", 2: "ground", 3: "interior"}


def main():
    for ds, (root, namef, gtf, ns) in CFG.items():
        I = {k: [] for k in NAME}
        be = []                       # interior depth-behind glass plane
        be_th = []                    # interior incidence (vs facade normal)
        for n in ns:
            d = f"{root}/{namef.format(n=n)}"
            files = sorted(glob.glob(f"{d}/frame_*.npz"))[::5]
            if not files:
                continue
            xyz = np.vstack([np.load(f)["xyz"] for f in files])
            inten = np.concatenate([np.load(f)["intensity"] for f in files])
            lab = np.concatenate([np.load(f)["label"] for f in files])
            gt = load_gt(gtf.format(n=n))
            nrm, e_u, e_v, O = plane_basis(gt)
            behind = -(xyz @ nrm + gt.plane_d)
            R = np.linalg.norm(xyz, axis=1)
            th = np.degrees(np.arccos(np.clip(np.abs(xyz @ nrm) / np.maximum(R, 1e-9), 0, 1)))
            for k in NAME:
                I[k].append(inten[lab == k])
            m = lab == 3
            be.append(behind[m]); be_th.append(th[m])
        I = {k: np.concatenate(v) if v else np.array([]) for k, v in I.items()}
        be = np.concatenate(be); be_th = np.concatenate(be_th)
        print(f"\n=== {ds.upper()} ===")
        for k, name in NAME.items():
            a = I[k]
            if a.size:
                print(f"  {name:9s}: n={a.size:7d}  I med={np.median(a):3.0f} "
                      f"p10={np.percentile(a,10):3.0f} p90={np.percentile(a,90):3.0f}  "
                      f"sat(>=254)={100*np.mean(a>=254):4.1f}%")
        if be.size:
            print(f"  INTERIOR depth behind glass plane (m): "
                  f"p10={np.percentile(be,10):.2f} p25={np.percentile(be,25):.2f} "
                  f"med={np.median(be):.2f} p75={np.percentile(be,75):.2f} "
                  f"p90={np.percentile(be,90):.2f} max={be.max():.1f}")
            # depth vs intensity correlation (does see-through dim with depth?)
            for lo, hi in [(0.1, 0.4), (0.4, 1.0), (1.0, 5.0)]:
                mm = (be >= lo) & (be < hi)
                if mm.sum() > 50:
                    intr_I = np.concatenate([I[3]])  # interior intensities align? approximate
            print(f"  INTERIOR incidence: med={np.median(be_th):.0f}deg p90={np.percentile(be_th,90):.0f}deg")


if __name__ == "__main__":
    main()
