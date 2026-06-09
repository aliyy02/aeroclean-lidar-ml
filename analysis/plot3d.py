"""3D matplotlib scatter of a labeled scan, colored by class (NED -> z-up view)."""
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {0: "gold", 1: "deepskyblue", 2: "limegreen", 3: "magenta"}
NAME = {0: "frame", 1: "glass", 2: "ground", 3: "interior"}


def load(ds, n):
    d = f"data/labeled_{ds}/{ds.capitalize()}_{n}"
    files = sorted(glob.glob(f"{d}/frame_*.npz"))
    xyz = np.vstack([np.load(f)["xyz"] for f in files])
    lab = np.concatenate([np.load(f)["label"] for f in files])
    return xyz, lab


def main():
    ds, n = sys.argv[1], int(sys.argv[2])
    maxpts = int(sys.argv[3]) if len(sys.argv) > 3 else 45000
    xyz, lab = load(ds, n)
    if len(xyz) > maxpts:
        idx = np.random.default_rng(0).choice(len(xyz), maxpts, replace=False)
        xyz, lab = xyz[idx], lab[idx]
    # NED (x fwd, y right, z down) -> view axes: X=right, Y=forward(depth), Z=up
    X, Y, Z = xyz[:, 1], xyz[:, 0], -xyz[:, 2]

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")
    for k in (2, 3, 0, 1):                       # ground, interior, frame, glass (glass on top)
        m = lab == k
        if m.any():
            ax.scatter(X[m], Y[m], Z[m], s=(10 if k == 1 else 2),
                       c=COL[k], label=f"{NAME[k]} ({m.sum()})",
                       alpha=(0.95 if k == 1 else 0.5), depthshade=False)
    ax.set_xlabel("right (m)"); ax.set_ylabel("forward / depth (m)"); ax.set_zlabel("up (m)")
    rng = [X.ptp(), Y.ptp(), Z.ptp()]
    ax.set_box_aspect(rng)
    ax.legend(markerscale=3, loc="upper left")
    ax.view_init(elev=16, azim=-72)
    ax.set_title(f"{ds} test {n} — labeled cloud (NED->z-up), {len(xyz)} pts")
    fig.tight_layout()
    out = f"analysis/cloud3d_{ds}_{n}.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}  ({dict((NAME[k], int((lab==k).sum())) for k in (0,1,2,3))})")


if __name__ == "__main__":
    main()
