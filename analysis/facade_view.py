"""Head-on facade view over a roll (alpha) sweep, to pin roll + y-sign from data.

Axis-maps native->NED as ned=(z, ysign*x, y) [the z-depth convention confirmed by
explore_raw], de-rolls about boresight x by each alpha, isolates the dense facade
shell (mode of depth +/- band), and plots the head-on (y, up=-z) scatter colored by
intensity with GT window rectangles overlaid. The alpha (and y-sign) where the
window pattern lines up with GT is the right one.

Usage (ROS sourced):
  PYTHONPATH=.:$PYTHONPATH python3 analysis/facade_view.py <bag> <gt> <out.png> \
      [--ysign 1|-1] [--alphas 0,12,20,24,28] [--color intensity|depth]
"""
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, ".")
from calibration.io_bag import read_all            # noqa: E402
from calibration.gt_parse import load_gt            # noqa: E402
from calibration.frames import apply_roll           # noqa: E402


def axis_map(xyz, ysign):
    out = np.empty_like(xyz)
    out[:, 0] = xyz[:, 2]              # depth = native z
    out[:, 1] = ysign * xyz[:, 0]      # y = +/- native x
    out[:, 2] = xyz[:, 1]              # down = native y
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag"); ap.add_argument("gt"); ap.add_argument("out")
    ap.add_argument("--ysign", type=int, default=1)
    ap.add_argument("--alphas", default="0,12,20,24,28")
    ap.add_argument("--color", default="intensity")
    ap.add_argument("--band", type=float, default=0.18)
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]

    gt = load_gt(a.gt)
    frames = read_all(a.bag)
    xyz = np.vstack([f.xyz for f in frames]).astype(float)
    inten = np.concatenate([f.intensity for f in frames]).astype(float)
    ned0 = axis_map(xyz, a.ysign)

    # GT window rects in (y, z)
    rects = []
    for w in gt.windows.values():
        ys = [c[1] for c in w.values()]; zs = [c[2] for c in w.values()]
        rects.append((min(ys), max(ys), min(zs), max(zs)))
    (gy0, gy1), (gz0, gz1) = gt.grid_bbox()

    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 6),
                             squeeze=False)
    for ax, al in zip(axes[0], alphas):
        ned = apply_roll(ned0, al)
        depth = ned[:, 0]
        # isolate dense facade shell: mode of depth within a window around GT
        lat = ((ned[:, 1] > gy0 - 0.3) & (ned[:, 1] < gy1 + 0.3) &
               (ned[:, 2] > gz0 - 0.3) & (ned[:, 2] < gz1 + 0.3))
        dd = depth[lat]
        h, e = np.histogram(dd, bins=120,
                            range=(gt.plane_x - 0.4, gt.plane_x + 0.7))
        mode = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
        sel = lat & (np.abs(depth - mode) < a.band)
        y, z = ned[sel, 1], ned[sel, 2]
        col = inten[sel] if a.color == "intensity" else (depth[sel] - gt.plane_x)
        sc = ax.scatter(y, -z, s=2, c=col, cmap="turbo",
                        vmin=np.percentile(col, 2), vmax=np.percentile(col, 98))
        for (u0, u1, v0, v1) in rects:
            ax.add_patch(Rectangle((u0, -v1), u1 - u0, v1 - v0, fill=False,
                                   ec="black", lw=1.3))
        ax.set_aspect("equal")
        ax.set_title(f"alpha={al:+.0f}  ysign={a.ysign:+d}\n"
                     f"shell@{mode:.3f} ({mode-gt.plane_x:+.3f} vs GT)  n={sel.sum()}")
        ax.set_xlabel("y (m, right)"); ax.set_ylabel("up = -z (m)")
        plt.colorbar(sc, ax=ax, shrink=0.7, label=a.color)
    fig.suptitle(f"{a.bag}  (GT plane_x={gt.plane_x:.3f})")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}  (n_frames={len(frames)}, pts={xyz.shape[0]})")


if __name__ == "__main__":
    main()
