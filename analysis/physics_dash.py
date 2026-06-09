"""Per-material physics dashboard for a real L2 facade dataset.

Aggregates many scans (pose-diverse, so incidence angle ranges 0..70+ deg) and
produces the figures that answer the forward-model questions:
  1 depth-behind-plane histogram, glass vs frame  (see-through / recess structure)
  2 intensity histogram, glass vs frame
  3 intensity vs incidence angle (binned median+IQR), glass & frame
  4 near-normal glass intensity vs range  (range-compensation test: flat => n~0)
  5 glass depth-behind vs incidence angle (recess=flat; see-through optical=varies)
  6 registration QA: fitted-front-normal vs GT-normal angle, per scan

Usage (ROS sourced):
  PYTHONPATH=.:$PYTHONPATH python3 analysis/physics_dash.py oxy|bech|l6 ALPHA YSIGN \
       OUT.png [all|frontal|near|far] [stride]
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from analysis.seg import segment, GLASS, FRAME           # noqa: E402
from calibration.io_bag import read_all                   # noqa: E402
from calibration.gt_parse import load_gt                  # noqa: E402

CFG = {
    "oxy":  ("Oxy_L2_All/Oxy_L2_test_{n}",
             "Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_{n}.txt", range(1, 60),
             {"near": range(31, 60), "far": range(1, 31), "frontal": [1, 11, 20, 31, 43]}),
    "bech": ("Bech_all/Bech_109_test_{n}",
             "Bech_Text_All/Bech_Text_All/Bech_109_test_{n}.txt", range(1, 41),
             {"near": range(1, 26), "far": range(26, 41), "frontal": [1, 26]}),
    "l6":   ("L6_all/L6_all/L6_test_{n}",
             "L6_tests_1/L6_tests_1/L6_test_{n}.txt", range(1, 44),
             {"frontal": [1, 9, 20]}),
}


def collect(ds, alpha, ysign, which, stride):
    bagt, gtt, allns, subsets = CFG[ds]
    ns = subsets.get(which, list(allns))
    G = {k: [] for k in ("th", "be", "bf", "I", "R")}
    F = {k: [] for k in ("th", "I", "be")}
    reg = []
    used = 0
    for n in ns:
        bag, gtp = bagt.format(n=n), gtt.format(n=n)
        if not os.path.isdir(bag):
            continue
        frames = read_all(bag)
        if stride > 1:
            frames = frames[::stride]
        gt = load_gt(gtp)
        s = segment(frames, gt, alpha, ysign)
        g = s.cls == GLASS
        f = s.cls == FRAME
        G["th"].append(np.degrees(s.theta_fit[g])); G["be"].append(s.behind_gt[g])
        G["bf"].append(s.behind_fit[g]); G["I"].append(s.intensity[g]); G["R"].append(s.R[g])
        F["th"].append(np.degrees(s.theta_fit[f])); F["I"].append(s.intensity[f])
        F["be"].append(s.behind_gt[f])
        ang = np.degrees(np.arccos(np.clip(abs(s.fit_normal @ gt.plane_normal /
              (np.linalg.norm(s.fit_normal) * np.linalg.norm(gt.plane_normal))), 0, 1)))
        reg.append((n, ang, float(np.median(s.behind_gt[g])) if g.any() else np.nan))
        used += 1
    G = {k: np.concatenate(v) if v else np.array([]) for k, v in G.items()}
    F = {k: np.concatenate(v) if v else np.array([]) for k, v in F.items()}
    print(f"{ds}/{which}: {used} scans, glass={G['I'].size}, frame={F['I'].size}")
    return G, F, reg


def binned(x, y, edges):
    med, q1, q3, ctr = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 30:
            med.append(np.median(y[m])); q1.append(np.percentile(y[m], 25))
            q3.append(np.percentile(y[m], 75)); ctr.append(0.5 * (a + b))
    return np.array(ctr), np.array(med), np.array(q1), np.array(q3)


def main():
    ds, alpha, ysign, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    which = sys.argv[5] if len(sys.argv) > 5 else "all"
    stride = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    G, F, reg = collect(ds, alpha, ysign, which, stride)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    # 1 depth-behind hist
    bins = np.linspace(-0.3, 0.8, 120)
    ax[0, 0].hist(G["be"], bins=bins, color="dodgerblue", alpha=0.6, label="glass", density=True)
    ax[0, 0].hist(F["be"], bins=bins, color="gold", alpha=0.6, label="frame", density=True)
    ax[0, 0].axvline(0, color="k", lw=1, ls="--")
    ax[0, 0].set_title("depth behind GT plane (m)"); ax[0, 0].legend()
    ax[0, 0].set_xlabel("+behind  ->")
    # 2 intensity hist
    ib = np.linspace(0, 256, 64)
    ax[0, 1].hist(G["I"], bins=ib, color="dodgerblue", alpha=0.6, label="glass", density=True)
    ax[0, 1].hist(F["I"], bins=ib, color="gold", alpha=0.6, label="frame", density=True)
    ax[0, 1].set_title("intensity"); ax[0, 1].legend(); ax[0, 1].set_xlabel("counts")
    # 3 intensity vs incidence
    edges = np.arange(0, 80, 5.0)
    for d, c, lab in [(G, "dodgerblue", "glass"), (F, "gold", "frame")]:
        ctr, med, q1, q3 = binned(d["th"], d["I"], edges)
        ax[0, 2].plot(ctr, med, "-o", color=c, label=lab)
        ax[0, 2].fill_between(ctr, q1, q3, color=c, alpha=0.2)
    ax[0, 2].set_title("intensity vs incidence angle"); ax[0, 2].legend()
    ax[0, 2].set_xlabel("theta (deg, vs fitted normal)"); ax[0, 2].set_ylabel("intensity")
    ax[0, 2].set_ylim(0, 256)
    # 4 near-normal glass intensity vs range
    nn = G["th"] < 12
    ax[1, 0].scatter(G["R"][nn], G["I"][nn], s=2, alpha=0.1, color="dodgerblue")
    redges = np.linspace(np.percentile(G["R"][nn], 1), np.percentile(G["R"][nn], 99), 12)
    ctr, med, q1, q3 = binned(G["R"][nn], G["I"][nn], redges)
    ax[1, 0].plot(ctr, med, "-o", color="k")
    ax[1, 0].set_title("near-normal (th<12) glass intensity vs range")
    ax[1, 0].set_xlabel("range R (m)"); ax[1, 0].set_ylabel("intensity"); ax[1, 0].set_ylim(0, 256)
    # 5 glass depth-behind vs incidence
    ctr, med, q1, q3 = binned(G["th"], G["bf"], edges)
    ax[1, 1].plot(ctr, med, "-o", color="purple")
    ax[1, 1].fill_between(ctr, q1, q3, color="purple", alpha=0.2)
    ax[1, 1].axhline(0, color="k", lw=1, ls="--")
    ax[1, 1].set_title("glass depth behind FITTED FRONT vs incidence")
    ax[1, 1].set_xlabel("theta (deg)"); ax[1, 1].set_ylabel("behind front (m)")
    # 6 registration QA
    ns = [r[0] for r in reg]; angs = [r[1] for r in reg]; offs = [r[2] for r in reg]
    ax[1, 2].bar(ns, angs, color="teal")
    ax[1, 2].set_title("fit-normal vs GT-normal angle (deg) per scan")
    ax[1, 2].set_xlabel("test #"); ax[1, 2].set_ylabel("deg")
    fig.suptitle(f"{ds}  alpha={alpha} ysign={ysign}  ({which})  "
                 f"glass median I={np.median(G['I']):.0f}, frame={np.median(F['I']):.0f}")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    # print key numbers
    print(f"glass: behind_gt med={np.median(G['be']):.3f} "
          f"p10={np.percentile(G['be'],10):.3f} p90={np.percentile(G['be'],90):.3f}")
    print(f"frame: behind_gt med={np.median(F['be']):.3f}")
    print(f"glass behind_fit med={np.median(G['bf']):.3f} "
          f"p10={np.percentile(G['bf'],10):.3f} p90={np.percentile(G['bf'],90):.3f}")
    reg_ang = np.array([r[1] for r in reg])
    print(f"registration fit-vs-GT normal: med={np.median(reg_ang):.1f} deg, "
          f"max={reg_ang.max():.1f} deg")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
