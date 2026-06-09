"""Catalogue a dataset's GT files: pose, standoff, incidence, window grid, sizes.

No ROS needed (parses GT .txt only). Helps pick scans that decouple range from
angle and confirms grid structure per dataset.

Usage: python3 analysis/catalog.py <gt_dir_glob>
  e.g. python3 analysis/catalog.py 'Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_*.txt'
"""
import sys
import glob
import re
import numpy as np

sys.path.insert(0, ".")
from calibration.gt_parse import load_gt    # noqa: E402


def num(p):
    m = re.search(r"_(\d+)\.txt$", p)
    return int(m.group(1)) if m else -1


def main():
    paths = sorted([p for p in glob.glob(sys.argv[1]) if "Zone.Identifier" not in p],
                   key=num)
    print(f"{len(paths)} GT files")
    print(f"{'test':>5} {'nwin':>4} {'standoff':>8} {'yaw':>6} {'pitch':>6} "
          f"{'roll':>6} {'planeX':>7} {'win_w':>6} {'win_h':>6} {'grid':>10}")
    standoffs, frontal = [], []
    for p in paths:
        try:
            gt = load_gt(p)
        except Exception as e:
            print(f"{num(p):5d}  parse-error: {e}")
            continue
        # standoff: distance from origin to plane along normal = |plane_d|
        standoff = abs(gt.plane_d)
        # window sizes from first window
        w0 = next(iter(gt.windows.values()))
        ww = np.linalg.norm(w0["UR"] - w0["UL"])
        wh = np.linalg.norm(w0["LL"] - w0["UL"])
        rows = set(); cols = set()
        for k in gt.windows:
            m = re.match(r"r(\d+)c(\d+)", k)
            if m:
                rows.add(int(m.group(1))); cols.add(int(m.group(2)))
        grid = f"{max(rows)}x{max(cols)}" if rows else "?"
        yaw, pit, rol = gt.pose["yaw"], gt.pose["pitch"], gt.pose["roll"]
        if abs(yaw) < 1 and abs(pit) < 1 and abs(rol) < 1:
            frontal.append(num(p))
        standoffs.append(standoff)
        print(f"{num(p):5d} {len(gt.windows):4d} {standoff:8.3f} {yaw:6.1f} "
              f"{pit:6.1f} {rol:6.1f} {gt.plane_x:7.3f} {ww:6.3f} {wh:6.3f} {grid:>10}")
    standoffs = np.array(standoffs)
    print(f"\nstandoff range: {standoffs.min():.3f}..{standoffs.max():.3f} m  "
          f"(median {np.median(standoffs):.3f})")
    print(f"frontal scans (|yaw,pitch,roll|<1): {sorted(frontal)}")


if __name__ == "__main__":
    main()
