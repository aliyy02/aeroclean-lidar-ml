"""Diagnose the labeling on a specific scan: where do glass/frame/interior go wrong?

Loads the labeled .npz we wrote (exact bag contents), recomputes incidence angle and
depth-behind, and asks:
  - is the FRAME (mullion) actually separated from glass, or all one class?
  - are the INTERIOR (purple) points real see-through, or bright high-incidence
    specular/glazing reflections off the panes (which would be high theta + high intensity)?

Usage: PYTHONPATH=.:$PYTHONPATH python3 analysis/diag_labels.py <ds> <N>
  ds in {oxy,bech,l6}; e.g. oxy 1   or   bech 7
"""
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, ".")
from calibration.gt_parse import load_gt                       # noqa: E402
from analysis.seg import plane_basis, window_rects_uv          # noqa: E402

GT = {"oxy": ("data/labeled_oxy/Oxy_{n}", "Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_{n}.txt"),
      "bech": ("data/labeled_bech/Bech_{n}", "Bech_Text_All/Bech_Text_All/Bech_109_test_{n}.txt"),
      "l6": ("data/labeled/L6_test_{n}", "L6_tests_1/L6_tests_1/L6_test_{n}.txt")}
NAME = {0: "not_glass", 1: "glass", 2: "ground", 3: "interior"}
COL = {0: "gray", 1: "deepskyblue", 2: "limegreen", 3: "magenta"}


def main():
    ds, n = sys.argv[1], int(sys.argv[2])
    ldir, gtp = GT[ds][0].format(n=n), GT[ds][1].format(n=n)
    gt = load_gt(gtp)
    xs, iss, ls = [], [], []
    for f in sorted(glob.glob(f"{ldir}/frame_*.npz")):
        d = np.load(f)
        xs.append(d["xyz"]); iss.append(d["intensity"]); ls.append(d["label"])
    xyz = np.vstack(xs); inten = np.concatenate(iss); lab = np.concatenate(ls)
    nrm, e_u, e_v, O = plane_basis(gt)
    dpl = gt.plane_d
    R = np.linalg.norm(xyz, axis=1)
    nP = xyz @ nrm
    behind = -(xyz @ nrm + dpl)
    theta = np.degrees(np.arccos(np.clip(np.abs(nP) / np.maximum(R, 1e-9), 0, 1)))
    ahead = nP < -1e-9
    t = np.where(ahead, -dpl / np.where(ahead, nP, 1.0), 0.0)
    Q = xyz * t[:, None]
    u = (Q - O) @ e_u; v = (Q - O) @ e_v
    rects = window_rects_uv(gt, e_u, e_v, O)

    print(f"\n=== {ds} test {n}  pose={gt.pose} ===")
    for k in (0, 1, 2, 3):
        m = lab == k
        if m.any():
            print(f"  {NAME[k]:9s}: {m.sum():7d} ({100*m.mean():4.1f}%)  "
                  f"I_med={np.median(inten[m]):3.0f}  theta_med={np.median(theta[m]):4.1f}  "
                  f"behind_med={np.median(behind[m]):+.2f}")
    gl = lab == 1; intr = lab == 3
    # KEY TEST: interior points -- real see-through (dim, low-theta) vs glazing ghost (bright, high-theta)?
    if intr.sum() > 50:
        hi = intr & (theta > 45)
        print(f"  INTERIOR high-incidence (theta>45): {hi.sum()} pts, I_med={np.median(inten[hi]) if hi.any() else 0:.0f}, "
              f"behind_med={np.median(behind[hi]) if hi.any() else 0:.2f}")
        lo = intr & (theta < 30)
        print(f"  INTERIOR low-incidence  (theta<30): {lo.sum()} pts, I_med={np.median(inten[lo]) if lo.any() else 0:.0f}, "
              f"behind_med={np.median(behind[lo]) if lo.any() else 0:.2f}")

    fig, ax = plt.subplots(2, 3, figsize=(19, 11))
    # back-project facade pts to plane for crisp view
    P = xyz.copy(); fac = np.isin(lab, [0, 1, 3]) & ahead
    P[fac] = xyz[fac] * t[fac, None]
    uu = (P - O) @ e_u; vv = (P - O) @ e_v
    for k in (2, 0, 1, 3):
        m = lab == k
        if m.any():
            ax[0, 0].scatter(uu[m], -vv[m], s=2, c=COL[k], label=f"{NAME[k]} ({m.sum()})", alpha=0.5)
    for (u0, u1, v0, v1) in rects:
        ax[0, 0].add_patch(Rectangle((u0, -v1), u1 - u0, v1 - v0, fill=False, ec="red", lw=1.3))
    ax[0, 0].set_title("by LABEL (red=GT windows)"); ax[0, 0].legend(markerscale=4, fontsize=8)
    ax[0, 0].set_aspect("equal")
    s1 = ax[0, 1].scatter(uu, -vv, s=2, c=inten, cmap="turbo", vmin=20, vmax=255)
    for (u0, u1, v0, v1) in rects:
        ax[0, 1].add_patch(Rectangle((u0, -v1), u1 - u0, v1 - v0, fill=False, ec="black", lw=1.0))
    ax[0, 1].set_title("by INTENSITY (see the bright frame?)"); ax[0, 1].set_aspect("equal")
    plt.colorbar(s1, ax=ax[0, 1], shrink=0.7)
    s2 = ax[0, 2].scatter(uu, -vv, s=2, c=np.clip(behind, -0.1, 1.0), cmap="turbo")
    ax[0, 2].set_title("by DEPTH behind GT plane"); ax[0, 2].set_aspect("equal")
    plt.colorbar(s2, ax=ax[0, 2], shrink=0.7)
    # intensity vs theta for glass and interior
    ax[1, 0].scatter(theta[gl], inten[gl], s=3, c="deepskyblue", alpha=0.2, label="glass")
    ax[1, 0].scatter(theta[intr], inten[intr], s=3, c="magenta", alpha=0.3, label="interior")
    ax[1, 0].set_xlabel("incidence theta (deg)"); ax[1, 0].set_ylabel("intensity")
    ax[1, 0].set_title("intensity vs incidence: glass vs interior"); ax[1, 0].legend()
    # depth vs theta for interior
    ax[1, 1].scatter(theta[intr], behind[intr], s=3, c=inten[intr], cmap="turbo", vmin=20, vmax=255)
    ax[1, 1].set_xlabel("incidence theta (deg)"); ax[1, 1].set_ylabel("behind plane (m)")
    ax[1, 1].set_title("INTERIOR: depth vs incidence (color=intensity)")
    # intensity histograms
    for k, c in ((1, "deepskyblue"), (0, "gray"), (3, "magenta")):
        m = lab == k
        if m.sum() > 20:
            ax[1, 2].hist(inten[m], bins=40, range=(0, 256), histtype="step", color=c,
                          label=NAME[k], density=True, lw=2)
    ax[1, 2].set_title("intensity by class"); ax[1, 2].legend(); ax[1, 2].set_xlabel("intensity")
    fig.suptitle(f"{ds} test {n}  pose={gt.pose}")
    fig.tight_layout(); out = f"analysis/diag_{ds}_{n}.png"
    fig.savefig(out, dpi=100); print(f"wrote {out}")


if __name__ == "__main__":
    main()
