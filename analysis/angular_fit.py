"""Fit forward_model.reflectance to each dataset's glass/frame I(theta); compare.

Tests whether ONE model form (a*((1-g)cos+g) + s*fresnel_grazing + b*burst) spans
L6 / Oxy / Bech. Uses reported intensity directly as rho (range-compensated, n~0).
Also shows depth-behind vs angle so surface vs see-through is visible alongside.

Usage (ROS sourced):
  PYTHONPATH=.:$PYTHONPATH python3 analysis/angular_fit.py [stride]
"""
import sys
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from analysis.agg import collect, binned_median            # noqa: E402
from forward_model.reflectance import reflectance          # noqa: E402

FINE = np.array([0, 2, 4, 6, 9, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
                float)


def fit_refl(th_deg, I_med):
    """Fit rho(theta)=a((1-g)cos+g)+s*fresnel+b*burst to (theta_deg, I) medians."""
    th = np.radians(th_deg)

    def resid(p):
        a, g, s, b, m = p
        return reflectance(th, a, g, s, b, m) - I_med
    p0 = [I_med.min(), 0.8, 20.0, max(I_med.max() - I_med.min(), 1.0), 0.12]
    lo = [0, 0, 0, 0, 0.01]; hi = [300, 1, 400, 400, 1.5]
    r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=20000)
    pred = reflectance(th, *r.x)
    ss_res = np.sum((pred - I_med) ** 2)
    ss_tot = np.sum((I_med - I_med.mean()) ** 2) + 1e-9
    return r.x, 1 - ss_res / ss_tot


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    data = {}
    for ds in ("l6", "oxy", "bech"):
        G, F, _ = collect(ds, which="all", stride=stride)
        data[ds] = (G, F)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    colors = {"l6": "seagreen", "oxy": "dodgerblue", "bech": "crimson"}
    fitinfo = {}
    for j, ds in enumerate(("l6", "oxy", "bech")):
        G, F = data[ds]
        cg, mg, q1g, q3g, ng = binned_median(G["th"], G["I"], FINE)
        cf, mf, q1f, q3f, nf = binned_median(F["th"], F["I"], FINE)
        pg, r2g = fit_refl(cg, mg)
        pf, r2f = fit_refl(cf, mf)
        fitinfo[ds] = (pg, r2g, pf, r2f)
        thx = np.linspace(0, 75, 200)
        ax[0, j].plot(cg, mg, "o", color="dodgerblue", label="glass data")
        ax[0, j].fill_between(cg, q1g, q3g, color="dodgerblue", alpha=0.15)
        ax[0, j].plot(thx, reflectance(np.radians(thx), *pg), "-", color="navy",
                      label=f"glass fit R2={r2g:.2f}")
        ax[0, j].plot(cf, mf, "s", color="goldenrod", label="frame data")
        ax[0, j].plot(thx, reflectance(np.radians(thx), *pf), "--", color="darkorange",
                      label=f"frame fit R2={r2f:.2f}")
        ax[0, j].set_title(f"{ds}: I(theta)"); ax[0, j].set_ylim(0, 270)
        ax[0, j].set_xlabel("incidence (deg)"); ax[0, j].set_ylabel("intensity")
        ax[0, j].legend(fontsize=8)
        # depth-behind vs angle (surface vs see-through)
        cb, mb, q1b, q3b, nb = binned_median(G["th"], G["bf"], FINE)
        ax[1, j].plot(cb, mb, "-o", color="purple", label="median depth")
        ax[1, j].fill_between(cb, q1b, q3b, color="purple", alpha=0.2)
        # p90 depth to show see-through tail
        p90 = []
        for a, b in zip(FINE[:-1], FINE[1:]):
            m = (G["th"] >= a) & (G["th"] < b)
            p90.append(np.percentile(G["bf"][m], 90) if m.sum() > 25 else np.nan)
        ax[1, j].plot(0.5 * (FINE[:-1] + FINE[1:]), p90, "--", color="red", label="p90 (tail)")
        ax[1, j].axhline(0, color="k", lw=0.8, ls=":")
        ax[1, j].set_title(f"{ds}: glass depth behind front vs angle")
        ax[1, j].set_xlabel("incidence (deg)"); ax[1, j].set_ylabel("behind front (m)")
        ax[1, j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("analysis/angular_fit.png", dpi=110)

    # comparison overlay
    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))
    for ds in ("l6", "oxy", "bech"):
        G, F = data[ds]
        cg, mg, *_ = binned_median(G["th"], G["I"], FINE)
        ax2[0].plot(cg, mg, "-o", color=colors[ds], label=f"{ds} glass")
        cf, mf, *_ = binned_median(F["th"], F["I"], FINE)
        ax2[1].plot(cf, mf, "-s", color=colors[ds], label=f"{ds} frame")
    ax2[0].set_title("GLASS intensity vs angle (3 datasets)")
    ax2[1].set_title("FRAME intensity vs angle (3 datasets)")
    for a in ax2:
        a.set_xlabel("incidence (deg)"); a.set_ylabel("intensity"); a.legend(); a.set_ylim(0, 270)
    fig2.tight_layout(); fig2.savefig("analysis/angular_compare.png", dpi=110)

    print("\n=== reflectance fits (a, g, s, b, m_deg | R2) ===")
    for ds in ("l6", "oxy", "bech"):
        pg, r2g, pf, r2f = fitinfo[ds]
        print(f"{ds:5s} glass: a={pg[0]:6.1f} g={pg[1]:.2f} s={pg[2]:6.1f} "
              f"b={pg[3]:6.1f} m={np.degrees(pg[4]):4.1f}deg | R2={r2g:.3f}")
        print(f"{ds:5s} frame: a={pf[0]:6.1f} g={pf[1]:.2f} s={pf[2]:6.1f} "
              f"b={pf[3]:6.1f} m={np.degrees(pf[4]):4.1f}deg | R2={r2f:.3f}")
    print("wrote analysis/angular_fit.png, analysis/angular_compare.png")


if __name__ == "__main__":
    main()
