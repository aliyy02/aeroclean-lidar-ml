"""Generate the slide-ready presentation figures into presentation/figures/.

Every figure is regenerated from the CLEAN labeled data (data/labeled*), in one
consistent style (big fonts, the RViz class palette). Filenames are slide-numbered
(Sxx_...) so they bind to DECK_PROMPT.md. Run from the ml_pipeline root:

    python3 analysis/make_slides.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

sys.path.insert(0, ".")
from forward_model.reflectance import reflectance          # noqa: E402
from forward_model.detect import return_probability        # noqa: E402

OUT = "presentation/figures"
os.makedirs(OUT, exist_ok=True)

# ---- style -------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 19, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.titleweight": "bold", "figure.dpi": 140, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25,
})
# class palette (matches RViz convention)
CLS = {0: "#E8A50C", 1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
CLS_NAME = {0: "frame", 1: "glass", 2: "ground", 3: "interior"}
# dataset palette
DS = {"L6": "#2ca02c", "Oxy": "#1f77b4", "Bech": "#d62728"}
DSROOT = {"L6": ("data/labeled", "L6_test_{n}"),
          "Oxy": ("data/labeled_oxy", "Oxy_{n}"),
          "Bech": ("data/labeled_bech", "Bech_{n}")}


def _frames(ds, n, stride=1):
    root, namef = DSROOT[ds]
    return sorted(glob.glob(f"{root}/{namef.format(n=n)}/frame_*.npz"))[::stride]


def load_one(ds, n, idx=None):
    """Load a single representative frame (densest if idx None)."""
    fs = _frames(ds, n)
    if not fs:
        return None
    if idx is None:
        # pick the frame with the most points (best coverage)
        idx = int(np.argmax([np.load(f)["xyz"].shape[0] for f in fs]))
    d = np.load(fs[min(idx, len(fs) - 1)])
    return dict(xyz=d["xyz"].astype(float), I=d["intensity"].astype(float),
                lab=d["label"].astype(int), nrm=d["normal"].astype(float))


def load_agg(ds, ns, stride=8):
    """Aggregate points over several bags/frames for statistics."""
    X, I, L, N = [], [], [], []
    for n in ns:
        for f in _frames(ds, n, stride):
            d = np.load(f)
            X.append(d["xyz"].astype(float)); I.append(d["intensity"].astype(float))
            L.append(d["label"].astype(int)); N.append(d["normal"].astype(float))
    if not X:
        return None
    return (np.vstack(X), np.concatenate(I), np.concatenate(L), np.vstack(N))


def incidence_deg(xyz, nrm):
    R = np.linalg.norm(xyz, axis=1)
    u = xyz / np.maximum(R, 1e-9)[:, None]
    c = np.abs(np.sum(u * nrm, axis=1))
    return np.degrees(np.arccos(np.clip(c, 0, 1)))


NS = {"L6": [1, 9, 20, 30, 43], "Oxy": [1, 11, 20, 31, 43], "Bech": [1, 14, 21, 26, 34]}


def stack_bag(ds, n, nframes=30):
    """Accumulate the first nframes of one bag (static rig pose) -> a dense cloud."""
    fs = _frames(ds, n)[:nframes]
    X = np.vstack([np.load(f)["xyz"].astype(float) for f in fs])
    I = np.concatenate([np.load(f)["intensity"].astype(float) for f in fs])
    L = np.concatenate([np.load(f)["label"].astype(int) for f in fs])
    N = np.vstack([np.load(f)["normal"].astype(float) for f in fs])
    return X, I, L, N


def facade_normal(lab, nrm):
    """One robust facade normal from the FRAME points (solid mullions = clean normals)."""
    sel = lab == 0
    if sel.sum() < 20:
        sel = (lab == 0) | (lab == 1)
    v = np.median(nrm[sel], axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def glass_angle_intensity(ds, ns, stride=2):
    """Per-frame: robust facade normal -> incidence for GLASS points. Returns (theta_deg, I)."""
    TH, II = [], []
    for n in ns:
        for f in _frames(ds, n, stride):
            d = np.load(f)
            xyz = d["xyz"].astype(float); I = d["intensity"].astype(float)
            lab = d["label"].astype(int); nrm = d["normal"].astype(float)
            if (lab == 1).sum() < 5:
                continue
            fn = facade_normal(lab, nrm)
            R = np.linalg.norm(xyz, axis=1)
            u = xyz / np.maximum(R, 1e-9)[:, None]
            th = np.degrees(np.arccos(np.clip(np.abs(u @ fn), 0, 1)))
            m = lab == 1
            TH.append(th[m]); II.append(I[m])
    if not TH:
        return np.array([]), np.array([])
    return np.concatenate(TH), np.concatenate(II)


# ============================================================ S08 point cloud explainer
def fig_pointcloud_explainer():
    """A REAL dense scan colored by intensity (windows visible) + plain text on what a
    point cloud is. No labels yet -- this is the raw thing the LiDAR gives us."""
    xyz, I, lab, N = stack_bag("L6", 1, 30)
    fig = plt.figure(figsize=(15, 6.2))
    ax = fig.add_subplot(121)
    s = ax.scatter(xyz[:, 1], -xyz[:, 2], c=I, cmap="turbo", s=2, vmin=0, vmax=255)
    gy = xyz[lab == 1, 1]; gz = -xyz[lab == 1, 2]
    ax.set_xlim(np.percentile(gy, 1) - 0.4, np.percentile(gy, 99) + 0.4)
    ax.set_ylim(np.percentile(gz, 1) - 0.6, np.percentile(gz, 99) + 0.4)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("y (m)"); ax.set_ylabel("height (m)")
    ax.set_title("A real LiDAR scan (raw, colored by intensity)")
    cb = fig.colorbar(s, ax=ax, shrink=0.8); cb.set_label("intensity (0–255)")

    ax2 = fig.add_subplot(122); ax2.axis("off")
    ax2.set_title("What is a point cloud?", loc="left")
    txt = ("Each laser beam that comes back = ONE point:\n\n"
           "   •  x, y, z   — where it hit, in metres\n"
           "   •  intensity — how bright the return was (0–255)\n"
           "   •  which laser ring, and when\n\n"
           "A scan is thousands of these points —\n"
           "unordered, no grid, no pixels.\n\n"
           "Look left: you can already SEE the window grid\n"
           "emerge from the points. The frames return\n"
           "cleanly; the glass is patchy because the beam\n"
           "mostly passes through it — that gap is the\n"
           "core difficulty of this project.")
    ax2.text(0.0, 0.95, txt, va="top", ha="left", fontsize=15.5, transform=ax2.transAxes)
    fig.tight_layout(); fig.savefig(f"{OUT}/S08_pointcloud_explainer.png"); plt.close(fig)


# ============================================================ S10a labeled scan
def _stack_bag(ds, n, nframes=20):
    """Accumulate the first nframes of one bag (static rig pose) for a dense frontal view."""
    fs = _frames(ds, n)[:nframes]
    X = np.vstack([np.load(f)["xyz"].astype(float) for f in fs])
    L = np.concatenate([np.load(f)["label"].astype(int) for f in fs])
    return X, L


def fig_labels():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, ds in zip(axes, ["L6", "Oxy", "Bech"]):
        xyz, lab = _stack_bag(ds, 1, 25)
        for k in [2, 0, 3, 1]:                      # draw order: ground, frame, interior, glass
            m = lab == k
            ax.scatter(xyz[m, 1], -xyz[m, 2], s=1.5, c=CLS[k], label=CLS_NAME[k])
        ax.set_title(f"{ds}  (real scan, labelled)")
        ax.set_xlabel("y  (m)"); ax.set_aspect("equal", "box")
        # zoom to the facade band (drop the far ground sweep / sparse outliers)
        gy = xyz[lab == 1, 1]; gz = -xyz[lab == 1, 2]
        if gy.size:
            ax.set_xlim(np.percentile(gy, 1) - 0.4, np.percentile(gy, 99) + 0.4)
            ax.set_ylim(np.percentile(gz, 1) - 0.6, np.percentile(gz, 99) + 0.4)
        ax.set_ylabel("height (m)" if ds == "L6" else "")
    h = [plt.Line2D([0], [0], marker="o", ls="", ms=9, color=CLS[k]) for k in [1, 0, 3, 2]]
    fig.legend(h, [CLS_NAME[k] for k in [1, 0, 3, 2]], loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ground-truth labelling: glass / frame / interior / ground",
                 fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(f"{OUT}/S10a_labels.png"); plt.close(fig)


# ============================================================ S10b class distribution
def fig_class_distribution():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    order = ["L6", "Oxy", "Bech"]
    bottoms = np.zeros(3)
    counts = {ds: np.zeros(4) for ds in order}
    for ds in order:
        agg = load_agg(ds, NS[ds])
        _, _, L, _ = agg
        for k in range(4):
            counts[ds][k] = np.mean(L == k) * 100
    for k in [0, 1, 2, 3]:
        vals = np.array([counts[ds][k] for ds in order])
        ax.bar(order, vals, bottom=bottoms, color=CLS[k], label=CLS_NAME[k],
               edgecolor="white")
        for i, (b, v) in enumerate(zip(bottoms, vals)):
            if v > 4:
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontweight="bold")
        bottoms += vals
    ax.set_ylabel("share of points (%)"); ax.set_ylim(0, 100)
    ax.set_title("Per-point class mix by dataset")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.savefig(f"{OUT}/S10b_class_distribution.png"); plt.close(fig)


# ============================================================ S12a angular (glass)
def fig_angular_glass():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    bins = np.arange(0, 72, 6); ctr = bins[:-1] + 3
    for ds in ["L6", "Oxy", "Bech"]:
        th, inten = glass_angle_intensity(ds, NS[ds])
        med, lo, hi = [], [], []
        for a, b in zip(bins[:-1], bins[1:]):
            s = inten[(th >= a) & (th < b)]
            if s.size >= 30:
                med.append(np.median(s)); lo.append(np.percentile(s, 25))
                hi.append(np.percentile(s, 75))
            else:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        med, lo, hi = np.array(med), np.array(lo), np.array(hi)
        ok = ~np.isnan(med)
        ax.fill_between(ctr[ok], lo[ok], hi[ok], color=DS[ds], alpha=0.15)
        ax.plot(ctr[ok], med[ok], "-o", color=DS[ds], lw=3, ms=7, label=f"{ds} glass")
    ax.axhline(255, ls="--", color="k", alpha=0.5)
    ax.text(50, 258, "sensor ceiling (255)", fontsize=11, color="#555")
    ax.set_xlabel("incidence angle  (0 = head-on  →  70 = grazing)")
    ax.set_ylabel("glass return intensity (0–255)")
    ax.set_title("How bright glass returns vs viewing angle")
    ax.annotate("L6: bright flash\nhead-on", (4, 232), (12, 190), color=DS["L6"], fontsize=12,
                arrowprops=dict(arrowstyle="->", color=DS["L6"]))
    ax.annotate("Bech: flares to 255\nat grazing", (60, 252), (30, 210), color=DS["Bech"],
                fontsize=12, arrowprops=dict(arrowstyle="->", color=DS["Bech"]))
    ax.legend(loc="center right"); ax.set_ylim(0, 275); ax.set_xlim(0, 70)
    fig.savefig(f"{OUT}/S12a_angular_glass.png"); plt.close(fig)


# ============================================================ S12b saturation
def fig_saturation():
    """Bech glass is ~87% pinned at the 255 ceiling, so a plain histogram hides the story
    (the spike runs off-chart). Show it explicitly: LEFT the shape of the UN-clipped returns,
    RIGHT the fraction pinned at 255."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.6))
    sat = {}
    for ds in ["L6", "Oxy", "Bech"]:
        agg = load_agg(ds, NS[ds]); _, I, L, _ = agg
        inten = I[L == 1]
        sat[ds] = 100 * np.mean(inten >= 254)
        unc = inten[inten < 254]
        a1.hist(unc, bins=np.arange(0, 256, 8), density=True, histtype="step", lw=3,
                color=DS[ds], label=f"{ds} glass")
    a1.set_xlabel("intensity of returns BELOW the ceiling (0–254)")
    a1.set_ylabel("fraction (normalised)")
    a1.set_title("Shape of the un-clipped glass returns")
    a1.legend(loc="upper left"); a1.set_xlim(0, 255)

    dss = ["L6", "Oxy", "Bech"]
    bars = a2.bar(dss, [sat[d] for d in dss], color=[DS[d] for d in dss], edgecolor="white")
    for d, b in zip(dss, bars):
        a2.text(b.get_x() + b.get_width() / 2, sat[d] + 2, f"{sat[d]:.0f}%",
                ha="center", fontsize=15, fontweight="bold")
    a2.set_ylabel("% of glass returns pinned at 255")
    a2.set_title("How often the glass SATURATES the sensor")
    a2.set_ylim(0, 100)
    fig.suptitle("Glass brightness clips at the sensor's 255 ceiling — Bechtel almost always",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{OUT}/S12b_saturation_hist.png")
    plt.close(fig)


# ============================================================ S12c see-through
def fig_seethrough():
    d = load_one("Bech", 14)
    xyz, lab = d["xyz"], d["lab"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for k in [0, 1, 3]:
        m = lab == k
        ax.scatter(xyz[m, 1], xyz[m, 0], s=4, c=CLS[k], label=CLS_NAME[k], alpha=0.7)
    gx = np.median(xyz[lab == 1, 0])
    ax.axhline(gx, ls="--", color="#1f77b4", lw=2)
    ax.text(ax.get_xlim()[0], gx, "  glass plane", color="#1f77b4", va="bottom", fontsize=13)
    ax.annotate("beams pass THROUGH the glass\nand return from inside the room",
                xy=(np.median(xyz[lab == 3, 1]), np.percentile(xyz[lab == 3, 0], 70)),
                xytext=(0.05, 0.92), textcoords="axes fraction", fontsize=14,
                color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_xlabel("y  — along the facade (m)"); ax.set_ylabel("x  — depth from sensor (m)")
    ax.set_title("Top-down view: glass is see-through (Bechtel)")
    ax.legend(loc="lower right")
    fig.savefig(f"{OUT}/S12c_seethrough.png"); plt.close(fig)


# ============================================================ S03 ransac baseline / why it breaks
def fig_ransac_baseline():
    """Both failure modes, on a REAL dense L6 bag: (1) many windows, not one board;
    (2) many depth layers, so one plane can't fit it."""
    xyz, I, lab, N = stack_bag("L6", 1, 30)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6))
    # left: frontal -> a GRID of windows
    for k in [2, 0, 3, 1]:
        m = lab == k
        a1.scatter(xyz[m, 1], -xyz[m, 2], s=1.5, c=CLS[k], label=CLS_NAME[k])
    gy = xyz[lab == 1, 1]; gz = -xyz[lab == 1, 2]
    a1.set_xlim(np.percentile(gy, 1) - 0.4, np.percentile(gy, 99) + 0.4)
    a1.set_ylim(np.percentile(gz, 1) - 0.6, np.percentile(gz, 99) + 0.4)
    a1.set_title("Problem 1 — a grid of windows")
    a1.set_xlabel("y (m)"); a1.set_ylabel("height (m)"); a1.set_aspect("equal", "box")
    a1.legend(loc="upper right", markerscale=4, fontsize=11)
    # right: top-down -> many depth layers; overlay the single plane RANSAC would pick
    for k in [0, 1, 3]:
        m = lab == k
        a2.scatter(xyz[m, 1], xyz[m, 0], s=2, c=CLS[k], label=CLS_NAME[k], alpha=0.6)
    gx = np.median(xyz[lab == 1, 0])
    a2.axhline(gx, ls="--", color="k", lw=2)
    a2.set_ylim(1.4, 4.6)
    a2.text(a2.get_xlim()[0], gx + 0.05, " the ONE plane RANSAC fits", va="bottom", fontsize=12)
    a2.set_title("Problem 2 — many depth layers")
    a2.set_xlabel("y (m)"); a2.set_ylabel("x — depth from sensor (m)")
    a2.legend(loc="upper right", markerscale=4, fontsize=11)
    fig.suptitle("Why the single-board, single-plane pipeline breaks on real windows",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/S03_ransac_baseline.png")
    plt.close(fig)


# ============================================================ S06 pointnet motivation
def fig_pointnet_motivation():
    """Clean schematic of PointNet++ set abstraction: sample centroids, group neighbours in a
    ball, pool to a feature, repeat at a coarser scale. Plain labels, no real-scan clutter."""
    rng = np.random.default_rng(7)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.8))

    pts = rng.uniform(0, 10, (160, 2))
    a1.scatter(pts[:, 0], pts[:, 1], s=14, c="#bbb", zorder=1)
    cen = np.array([[2.5, 7.0], [6.8, 7.6], [3.2, 3.0], [7.4, 2.8]])
    for c in cen:
        a1.add_patch(Circle(c, 1.6, fill=True, fc="#1f77b4", ec="#1f77b4", alpha=0.12, zorder=2))
        a1.add_patch(Circle(c, 1.6, fill=False, ec="#1f77b4", lw=2, zorder=3))
        a1.scatter(*c, s=90, c="#d62728", marker="*", zorder=4)
    a1.scatter([], [], s=90, c="#d62728", marker="*", label="sampled centroid")
    a1.add_patch(Circle((-9, -9), 0.1, ec="#1f77b4", fc="none", label="grouping ball"))
    a1.set_title("Set abstraction: group nearby points,\npool each ball into one feature")
    a1.set_xlim(0, 10); a1.set_ylim(0, 10); a1.set_aspect("equal", "box")
    a1.set_xticks([]); a1.set_yticks([]); a1.legend(loc="lower right", fontsize=12)

    a2.axis("off"); a2.set_title("Why PointNet++?", loc="left")
    txt = ("The old pipeline leaned on two cues — how DENSE the\n"
           "points are, and the local GEOMETRY — baked into\n"
           "hand-tuned thresholds. Glass violates all of them.\n\n"
           "PointNet++ LEARNS those same cues:\n\n"
           "   • picks centre points across the cloud\n"
           "   • groups each one's neighbours in a ball\n"
           "   • pools each ball into a local feature\n"
           "   • repeats at coarser and coarser scales\n"
           "   • works directly on unordered points\n\n"
           "→ same signals, but learned — and robust to glass,\n"
           "   many windows, and oblique views.")
    a2.text(0.0, 0.97, txt, va="top", fontsize=15.5, transform=a2.transAxes)
    fig.tight_layout(); fig.savefig(f"{OUT}/S06_pointnet_motivation.png"); plt.close(fig)


# ============================================================ S07 test-bed schematic
def fig_testbed_schematic():
    fig, ax = plt.subplots(figsize=(11, 6.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    # facade with windows
    ax.add_patch(Rectangle((7.2, 0.6), 2.2, 5.6, fc="#d9e6f2", ec="k", lw=2))
    for r in range(3):
        for c in range(2):
            ax.add_patch(Rectangle((7.45 + c * 1.0, 1.1 + r * 1.7), 0.8, 1.3,
                                   fc="#1f77b4", ec="k", alpha=0.6))
    ax.text(8.3, 6.4, "glass facade\n(known corners = GT)", ha="center", fontsize=13)
    # rig + lidar
    ax.add_patch(Rectangle((1.0, 3.0), 0.9, 0.9, fc="#333", ec="k"))
    ax.text(1.45, 2.7, "Unitree L2", ha="center", fontsize=12)
    ax.text(1.45, 4.2, "6-DOF rig\n(CNC-like)", ha="center", fontsize=13, fontweight="bold")
    # beams
    rng = np.random.default_rng(0)
    for zt in np.linspace(1.0, 6.0, 11):
        ax.plot([1.9, 7.2], [3.45, zt], color="#E8A50C", lw=0.8, alpha=0.7)
    # pose arrows
    for (dx, dy, lab) in [(0, 1, "z"), (1, 0, "x"), (0.6, 0.6, "yaw/pitch/roll")]:
        ax.add_patch(FancyArrowPatch((1.45, 3.45), (1.45 + dx, 3.45 + dy),
                     arrowstyle="->", mutation_scale=16, color="#2ca02c"))
    ax.text(3.4, 6.3, "rig commands a KNOWN 6-DOF pose → every scan has\n"
                      "ground-truth sensor position + window-corner geometry",
            fontsize=13, ha="left")
    ax.set_title("Physical data gathering: 6-DOF LiDAR test bed", fontsize=19, fontweight="bold")
    fig.savefig(f"{OUT}/S07_testbed_schematic.png"); plt.close(fig)


# ============================================================ S11 sim-to-real gap
def fig_sim_to_real():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))
    # synthetic: clean dense grid, every beam returns, uniform intensity
    yy, zz = np.meshgrid(np.linspace(-0.8, 0.8, 70), np.linspace(-0.6, 0.6, 52))
    a1.scatter(yy.ravel(), zz.ravel(), s=7, c="#1f77b4")
    a1.set_title("Simulator: every beam returns, uniform")
    a1.set_xlabel("y (m)"); a1.set_ylabel("height (m)"); a1.set_aspect("equal", "box")
    # real: a DENSE real scan colored by intensity (all facade points) -> sparse glass + saturation
    xyz, I, lab, N = stack_bag("L6", 1, 30)
    keep = lab != 2                       # drop ground; keep frame+glass+interior
    gy = xyz[lab == 1, 1]; gz = -xyz[lab == 1, 2]
    sc = a2.scatter(xyz[keep, 1], -xyz[keep, 2], s=2, c=I[keep], cmap="turbo", vmin=0, vmax=255)
    a2.set_xlim(np.percentile(gy, 1) - 0.4, np.percentile(gy, 99) + 0.4)
    a2.set_ylim(np.percentile(gz, 1) - 0.6, np.percentile(gz, 99) + 0.4)
    a2.set_title("Real LiDAR: patchy, holes, clipped at 255")
    a2.set_xlabel("y (m)"); a2.set_aspect("equal", "box")
    fig.colorbar(sc, ax=a2, shrink=0.8, label="intensity (0–255)")
    fig.suptitle("The sim-to-real gap the forward model must close",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/S11_sim_to_real_gap.png")
    plt.close(fig)


# ============================================================ S13 forward model
def fig_forward_model():
    """Plain-language schematic (no jargon): LEFT the three ways a surface returns the beam
    (the model just adds them up); RIGHT the three real glasses as different mixes."""
    th = np.linspace(0, 75, 200)
    spike = np.exp(-(th / 6.0) ** 2)
    graze = (th / 75.0) ** 3.0
    L6 = 85 + 170 * spike + 18 * graze
    Oxy = 130 + 33 * np.exp(-(th / 9.0) ** 2) + 40 * (th / 75.0) ** 2.5
    Bech = np.clip(65 + 55 * np.exp(-(th / 6.0) ** 2) + 200 * (th / 72.0) ** 3.5, 0, 255)
    fig, (a, b) = plt.subplots(1, 2, figsize=(15.5, 6.2))
    a.plot(th, 90 + 0 * th, lw=3.5, color=DS["L6"],
           label="1) MATTE (wall): same at every angle")
    a.plot(th, 250 * spike, lw=3.5, color="#1f77b4",
           label="2) MIRROR (shiny): only head-on")
    a.plot(th, 250 * graze, lw=3.5, color=DS["Bech"],
           label="3) GLASS edge-on: flares near grazing")
    a.set_title("The 3 ways a surface sends the laser back\n(the model just adds these up)")
    a.set_xlabel("how slanted the beam hits  (0 = straight on → 75 = edge-on)")
    a.set_ylabel("light sent back to the LiDAR")
    a.legend(loc="upper center"); a.set_ylim(0, 272)
    b.plot(th, L6, lw=3.5, color=DS["L6"], label="L6 = mirror flash + steady glow")
    b.plot(th, Oxy, lw=3.5, color="#1f77b4", label="Oxy = mostly steady glow (flat)")
    b.plot(th, Bech, lw=3.5, color=DS["Bech"], label="Bech = dark, only flares edge-on")
    b.axhline(255, ls="--", color="k", alpha=0.5); b.text(2, 258, "clips at 255", fontsize=11)
    b.set_title("The three real glasses = different mixes")
    b.set_xlabel("how slanted the beam hits  (0 = straight on → 75 = edge-on)")
    b.set_ylabel("brightness that comes back (0–255)")
    b.set_ylim(0, 272); b.legend(loc="upper center")
    b.annotate("dark head-on\n(tint absorbs the laser)", (8, 66), (22, 100),
               color=DS["Bech"], fontsize=12, arrowprops=dict(arrowstyle="->", color=DS["Bech"]))
    fig.suptitle("Forward model: how each surface returns the laser (calibrated to real scans)",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{OUT}/S13_forward_model.png")
    plt.close(fig)


def fig_cloud3d():
    """Hero real-scan view: one dense labelled L6 cloud in 3-D — the window grid is obvious."""
    xyz, I, lab, N = stack_bag("L6", 1, 30)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    for k in [2, 0, 3, 1]:
        m = lab == k
        ax.scatter(xyz[m, 1], xyz[m, 0], -xyz[m, 2], s=2, c=CLS[k], label=CLS_NAME[k])
    ax.set_xlabel("y — along facade (m)"); ax.set_ylabel("x — depth (m)")
    ax.set_zlabel("height (m)")
    ax.view_init(elev=14, azim=-78)
    ax.set_title("A real labelled scan (L6): the window grid in the point cloud")
    ax.legend(loc="upper left", markerscale=4, fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/S10c_cloud3d.png"); plt.close(fig)


# ============================================================ S16 results placeholder
def fig_results_placeholder():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))
    names = ["glass", "frame", "interior", "ground"]
    cm = np.full((4, 4), np.nan)
    a1.imshow(np.zeros((4, 4)), cmap="Greys", vmin=0, vmax=1)
    a1.set_xticks(range(4)); a1.set_yticks(range(4))
    a1.set_xticklabels(names, rotation=30); a1.set_yticklabels(names)
    for i in range(4):
        for j in range(4):
            a1.text(j, i, "?", ha="center", va="center", fontsize=22, color="#999")
    a1.set_title("Confusion matrix\n(PENDING INFERENCE)")
    a1.set_xlabel("predicted"); a1.set_ylabel("true"); a1.grid(False)

    a2.axis("off"); a2.set_title("Metrics — to fill after inference")
    rows = [["mIoU (4-class)", "—"], ["glass IoU", "—"], ["per-class F1", "—"],
            ["corner error (cm)", "—"], ["inference rate (Hz, RPi5)", "—"],
            ["sim→real transfer Δ", "—"]]
    tbl = a2.table(cellText=rows, colLabels=["metric", "value"],
                   loc="center", cellLoc="left", colWidths=[0.6, 0.3])
    tbl.auto_set_font_size(False); tbl.set_fontsize(14); tbl.scale(1, 2.0)
    fig.suptitle("Evaluation (model trained; inference pending)",
                 fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{OUT}/S16_results_placeholder.png")
    plt.close(fig)


FIGS = [fig_pointcloud_explainer, fig_labels, fig_cloud3d, fig_class_distribution,
        fig_angular_glass, fig_saturation, fig_seethrough, fig_ransac_baseline,
        fig_pointnet_motivation, fig_testbed_schematic, fig_sim_to_real, fig_forward_model,
        fig_results_placeholder]


if __name__ == "__main__":
    for f in FIGS:
        try:
            f(); print(f"OK  {f.__name__}")
        except Exception as e:                                   # noqa: BLE001
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print("figures in", OUT)
