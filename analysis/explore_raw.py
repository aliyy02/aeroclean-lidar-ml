"""Raw peek + axis-map brute force for a real L2 bag, from scratch.

Usage (ROS sourced):
    PYTHONPATH=.:$PYTHONPATH python3 analysis/explore_raw.py <bag_dir> <gt.txt>

Reports native xyz/intensity stats, then brute-forces all 48 signed axis
permutations native->NED and scores each by how well the dense facade lands on
the GT plane (correct mapping = a tight wall at the GT depth within the grid).
Makes NO assumption about the L6 (sz,sx,sy) convention.
"""
import sys
import itertools
import numpy as np

sys.path.insert(0, ".")
from calibration.io_bag import read_all          # noqa: E402
from calibration.gt_parse import load_gt          # noqa: E402


def stats(name, a):
    print(f"  {name:10s} min={a.min():8.3f} max={a.max():8.3f} "
          f"mean={a.mean():8.3f} std={a.std():7.3f}")


def main():
    bag, gtp = sys.argv[1], sys.argv[2]
    gt = load_gt(gtp)
    frames = read_all(bag)
    xyz = np.vstack([f.xyz for f in frames]).astype(float)
    inten = np.concatenate([f.intensity for f in frames]).astype(float)
    n = xyz.shape[0]
    print(f"\n=== {bag} ===")
    print(f"frames={len(frames)}  total_points={n}  "
          f"pts/frame={n/max(len(frames),1):.0f}")
    print(f"GT: plane_x={gt.plane_x:.4f} m  normal={gt.plane_normal}  "
          f"pose={gt.pose}")
    (gy0, gy1), (gz0, gz1) = gt.grid_bbox()
    print(f"GT grid bbox: y[{gy0:.3f},{gy1:.3f}] z[{gz0:.3f},{gz1:.3f}]  "
          f"(W={gy1-gy0:.3f} H={gz1-gz0:.3f})")

    print("native xyz ranges (meters):")
    stats("x", xyz[:, 0]); stats("y", xyz[:, 1]); stats("z", xyz[:, 2])
    R = np.linalg.norm(xyz, axis=1)
    stats("range", R)
    print(f"intensity: min={inten.min():.1f} max={inten.max():.1f} "
          f"mean={inten.mean():.1f}  unique={len(np.unique(inten))}")
    qs = np.percentile(inten, [0, 10, 25, 50, 75, 90, 99, 100])
    print("  intensity pct[0,10,25,50,75,90,99,100]=",
          np.round(qs, 1))

    # ---- brute force 48 signed axis permutations ----
    px = gt.plane_x
    m = 1.0                                   # generous lateral margin (roll-tolerant)
    rows = []
    for perm in itertools.permutations(range(3)):
        for sx, sy, sz in itertools.product((1, -1), repeat=3):
            ned = np.empty_like(xyz)
            ned[:, 0] = sx * xyz[:, perm[0]]
            ned[:, 1] = sy * xyz[:, perm[1]]
            ned[:, 2] = sz * xyz[:, perm[2]]
            depth = ned[:, 0]
            near = np.abs(depth - px) < 0.15
            lat = ((ned[:, 1] > gy0 - m) & (ned[:, 1] < gy1 + m) &
                   (ned[:, 2] > gz0 - m) & (ned[:, 2] < gz1 + m))
            sel = near & lat
            cnt = int(sel.sum())
            rows.append((cnt, perm, (sx, sy, sz), ned))
    rows.sort(key=lambda r: -r[0])
    print(f"\nTop axis mappings (near GT plane_x={px:.3f}, within grid+{m}m):")
    print("  rank  count   perm        signs      facade y/z extents (of near pts)")
    for i, (cnt, perm, sgn, ned) in enumerate(rows[:6]):
        depth = ned[:, 0]
        near = np.abs(depth - px) < 0.15
        ny, nz = ned[near, 1], ned[near, 2]
        ext = (f"y[{ny.min():.2f},{ny.max():.2f}] z[{nz.min():.2f},{nz.max():.2f}]"
               if near.any() else "(none)")
        axislbl = "".join("xyz"[p] for p in perm)
        print(f"  {i:4d}  {cnt:6d}  ned=({axislbl}) {str(sgn):12s} {ext}")

    # depth histogram for the winning mapping, to see panel structure
    cnt, perm, sgn, ned = rows[0]
    depth = ned[:, 0]
    lat = ((ned[:, 1] > gy0 - 0.3) & (ned[:, 1] < gy1 + 0.3) &
           (ned[:, 2] > gz0 - 0.3) & (ned[:, 2] < gz1 + 0.3))
    dd = depth[lat]
    print(f"\nWinning mapping depth histogram (lateral-gated), GT plane_x={px:.3f}:")
    lo, hi = px - 0.4, px + 0.6
    h, edges = np.histogram(dd[(dd > lo) & (dd < hi)], bins=40)
    for c, e0, e1 in zip(h, edges[:-1], edges[1:]):
        bar = "#" * int(60 * c / max(h.max(), 1))
        mark = "  <-- GT" if e0 <= px < e1 else ""
        print(f"  {e0:6.3f}..{e1:6.3f} {c:6d} {bar}{mark}")


if __name__ == "__main__":
    main()
