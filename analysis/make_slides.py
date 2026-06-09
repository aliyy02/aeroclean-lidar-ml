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


# ============================================================ S08 point cloud explainer
def fig_pointcloud_explainer():
    d = load_one("Oxy", 1)
    xyz, I = d["xyz"], d["I"]
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(121, projection="3d")
    s = ax.scatter(xyz[:, 1], xyz[:, 0], -xyz[:, 2], c=I, cmap="viridis",
                   s=3, vmin=0, vmax=255)
    ax.set_xlabel("y  (horizontal, m)"); ax.set_ylabel("x  (depth, m)")
    ax.set_zlabel("height, m"); ax.set_title("One LiDAR frame, colored by intensity")
    ax.view_init(elev=18, azim=-72)
    cb = fig.colorbar(s, ax=ax, shrink=0.6, pad=0.02); cb.set_label("intensity (0–255)")

    ax2 = fig.add_subplot(122); ax2.axis("off")
    ax2.set_title("What is a point cloud?")
    txt = ("Each LiDAR beam that returns gives ONE point:\n\n"
           "   •  x, y, z   — 3-D position (metres)\n"
           "   •  intensity — strength of the return (0–255)\n"
           "   •  ring, time — which laser, when\n\n"
           f"This frame:  {xyz.shape[0]:,} points\n\n"
           "Unordered + unstructured: no grid, no pixels.\n"
           "Density varies with range & surface. Glass mostly\n"
           "lets the beam THROUGH, so it returns few, erratic\n"
           "points — the core difficulty of this project.")
    ax2.text(0.02, 0.95, txt, va="top", ha="left", fontsize=15, family="monospace",
             transform=ax2.transAxes)
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
    fig, ax = plt.subplots(figsize=(9.5, 6))
    bins = np.arange(0, 75, 5)
    ctr = bins[:-1] + 2.5
    for ds in ["L6", "Oxy", "Bech"]:
        agg = load_agg(ds, NS[ds])
        X, I, L, N = agg
        m = L == 1
        th = incidence_deg(X[m], N[m]); inten = I[m]
        med = [np.median(inten[(th >= a) & (th < b)]) if np.any((th >= a) & (th < b)) else np.nan
               for a, b in zip(bins[:-1], bins[1:])]
        ax.plot(ctr, med, "-o", color=DS[ds], lw=2.5, ms=7, label=f"{ds} glass")
    ax.axhline(255, ls="--", color="k", alpha=0.5)
    ax.text(2, 248, "sensor ceiling (255)", fontsize=12)
    ax.set_xlabel("incidence angle (deg)"); ax.set_ylabel("median intensity (0–255)")
    ax.set_title("Glass return intensity vs incidence — three glass types")
    ax.legend(); ax.set_ylim(0, 270)
    fig.savefig(f"{OUT}/S12a_angular_glass.png"); plt.close(fig)


# ============================================================ S12b saturation
def fig_saturation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, ds in zip(axes, ["L6", "Oxy", "Bech"]):
        agg = load_agg(ds, NS[ds])
        _, I, L, _ = agg
        inten = I[L == 1]
        sat = 100 * np.mean(inten >= 254)
        ax.hist(inten, bins=np.arange(0, 257, 8), color=DS[ds], alpha=0.85)
        ax.axvline(255, ls="--", color="k")
        ax.set_title(f"{ds} glass\n{sat:.0f}% saturated at 255")
        ax.set_xlabel("intensity")
    axes[0].set_ylabel("point count")
    fig.suptitle("Glass intensity histograms — saturation clipping at 255",
                 fontsize=19, fontweight="bold")
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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))
    # left: ideal single board -> one tight plane
    rng = np.random.default_rng(0)
    yb = rng.uniform(-0.8, 0.8, 4000); zb = rng.uniform(-0.5, 0.5, 4000)
    xb = 2.0 + rng.normal(0, 0.01, 4000)
    a1.scatter(xb, zb, s=3, c="#555")
    a1.axvline(2.0, ls="--", color="#2ca02c", lw=2)
    a1.set_title("Single flat board\n→ ONE plane, RANSAC works")
    a1.set_xlabel("x — depth (m)"); a1.set_ylabel("height (m)")
    a1.set_xlim(1.6, 3.4)
    a1.text(2.02, 0.45, "RANSAC plane", color="#2ca02c", fontsize=12)
    # right: real facade -> multiple depth layers
    d = load_one("Bech", 14)
    xyz, lab = d["xyz"], d["lab"]
    for k in [0, 1, 3]:
        m = lab == k
        a2.hist(xyz[m, 0], bins=np.arange(1.6, 3.4, 0.04), color=CLS[k],
                alpha=0.7, label=CLS_NAME[k])
    a2.set_title("Real window facade\n→ MANY depth layers, single plane fails")
    a2.set_xlabel("x — depth (m)"); a2.set_ylabel("point count")
    a2.set_xlim(1.6, 3.4); a2.legend()
    fig.suptitle("Why the geometric (RANSAC) baseline breaks on windows",
                 fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{OUT}/S03_ransac_baseline.png")
    plt.close(fig)


# ============================================================ S06 pointnet motivation
def fig_pointnet_motivation():
    d = load_one("Oxy", 1)
    xyz = d["xyz"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))
    y, z = xyz[:, 1], -xyz[:, 2]
    a1.scatter(y, z, s=2, c="#888")
    # draw a few query balls (local neighborhoods)
    rng = np.random.default_rng(2)
    for _ in range(6):
        i = rng.integers(len(y))
        a1.add_patch(Circle((y[i], z[i]), 0.28, fill=False, ec="#d62728", lw=2))
    a1.set_title("PointNet++ learns LOCAL structure\n(query balls → features)")
    a1.set_xlabel("y (m)"); a1.set_ylabel("height (m)"); a1.set_aspect("equal", "box")

    a2.axis("off"); a2.set_title("Why PointNet++ ?")
    txt = ("RANSAC needs hand-tuned geometry:\n"
           "  plane thresholds, point density, corner walks.\n"
           "Glass breaks every one of those assumptions.\n\n"
           "PointNet++ LEARNS the same cues automatically:\n"
           "  • samples local neighbourhoods (query balls)\n"
           "  • pools them into hierarchical features\n"
           "  • captures density + geometric structure\n"
           "  • works directly on unordered points\n\n"
           "→ same signals RANSAC used, but learned\n"
           "   and robust to glass / multi-window / oblique.")
    a2.text(0.02, 0.95, txt, va="top", fontsize=15, family="monospace",
            transform=a2.transAxes)
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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))
    # synthetic: clean dense grid, every beam returns, uniform intensity
    yy, zz = np.meshgrid(np.linspace(-0.8, 0.8, 60), np.linspace(-0.6, 0.6, 45))
    a1.scatter(yy.ravel(), zz.ravel(), s=6, c="#1f77b4")
    a1.set_title("Simulator (raw geometry)\nevery beam returns, no glass physics")
    a1.set_xlabel("y (m)"); a1.set_ylabel("height (m)"); a1.set_aspect("equal", "box")
    # real: sparse + saturated glass
    d = load_one("Bech", 14)
    xyz, lab, I = d["xyz"], d["lab"], d["I"]
    m = lab == 1
    sc = a2.scatter(xyz[m, 1], -xyz[m, 2], s=6, c=I[m], cmap="inferno", vmin=0, vmax=255)
    a2.set_title("Real LiDAR on glass\nsparse, erratic, saturated returns")
    a2.set_xlabel("y (m)"); a2.set_aspect("equal", "box")
    fig.colorbar(sc, ax=a2, shrink=0.7, label="intensity")
    fig.suptitle("The sim-to-real gap the forward model must close",
                 fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{OUT}/S11_sim_to_real_gap.png")
    plt.close(fig)


# ============================================================ S13 forward model
def fig_forward_model():
    th = np.radians(np.linspace(0, 75, 200))
    archetypes = {
        "L6  coated-specular": dict(a=80, g=0.7, s=20, b=120, m=0.10, c="#2ca02c"),
        "Oxy coated-diffuse":  dict(a=130, g=0.95, s=15, b=8, m=0.15, c="#1f77b4"),
        "Bech tinted-Fresnel": dict(a=70, g=0.4, s=190, b=8, m=0.15, c="#d62728"),
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))
    for name, p in archetypes.items():
        rho = reflectance(th, p["a"], p["g"], p["s"], p["b"], p["m"])
        a1.plot(np.degrees(th), np.minimum(rho, 255), color=p["c"], lw=2.5, label=name)
    a1.axhline(255, ls="--", color="k", alpha=0.5); a1.text(2, 258, "clip at 255", fontsize=11)
    a1.set_title("Reflectance model  ρ(θ)\n(diffuse + grazing + specular burst)")
    a1.set_xlabel("incidence angle (deg)"); a1.set_ylabel("reported intensity")
    a1.legend(fontsize=12); a1.set_ylim(0, 275)

    for pf, lab, col in [(1.0, "matte / coated  (p_floor=1)", "#2ca02c"),
                         (0.05, "mirror-like  (p_floor≈0)", "#d62728"),
                         (0.4, "tinted  (p_floor=0.4)", "#1f77b4")]:
        pr = return_probability(th, pf, cone=np.radians(12))
        a2.plot(np.degrees(th), pr, color=col, lw=2.5, label=lab)
    a2.set_title("Return-probability model  P(θ)\ndoes the beam come back at all?")
    a2.set_xlabel("incidence angle (deg)"); a2.set_ylabel("P(return)")
    a2.legend(fontsize=12); a2.set_ylim(0, 1.05)
    fig.suptitle("First-principles forward model (range-independent ρ; 1/R² only in detection)",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{OUT}/S13_forward_model.png")
    plt.close(fig)


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


FIGS = [fig_pointcloud_explainer, fig_labels, fig_class_distribution, fig_angular_glass,
        fig_saturation, fig_seethrough, fig_ransac_baseline, fig_pointnet_motivation,
        fig_testbed_schematic, fig_sim_to_real, fig_forward_model, fig_results_placeholder]


if __name__ == "__main__":
    for f in FIGS:
        try:
            f(); print(f"OK  {f.__name__}")
        except Exception as e:                                   # noqa: BLE001
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print("figures in", OUT)
