"""Return probability done right: divide OUT the L2 beam pattern, then fit the noise floor.

The L2 is pitched so its spin axis points at the wall, so beam density per steradian
FALLS as the angle from boresight (phi) grows -- exactly the user's point. On a frontal
wall phi == incidence angle, and the wide panel + offset boresight gives phi up to ~70deg
in a single frontal scan. So:

  returns_per_steradian(class, phi) = n(phi) * P_return(class)
     where n(phi) = the lidar beam pattern (same for every material at that phi).

Measuring an OPAQUE reference (facade wall+frame, which returns every beam, P~1) gives n(phi);
the window curve divided by it is P_return for a through-window beam. We get the per-class
solid angle in each phi-ring analytically from GT (sample azimuth psi around boresight).

Then the noise floor: a beam is kept iff received power C*I/R^2 > T, i.e. the faintest
DETECTED intensity rises as I_min ~ (T/C)*R^2. Fitting the low-percentile intensity vs range
of the dim returns recovers T/C.

Usage (ROS sourced):
  PYTHONPATH=.:$PYTHONPATH python3 analysis/return_prob2.py [stride]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from analysis.seg import segment, GLASS, plane_basis, window_rects_uv   # noqa: E402
from analysis.agg import CFG, collect                                   # noqa: E402
from calibration.io_bag import read_all                                 # noqa: E402
from calibration.gt_parse import load_gt                                # noqa: E402

PHI = np.arange(2.5, 72, 5.0)             # incidence/phi bin centers (deg)
DPHI = np.radians(5.0)


def gt_window_fraction(gt, npsi=3000):
    """Fraction of each phi-ring (around boresight, which hits the facade at y=z=0)
    that falls inside a window aperture. Frontal facade at depth D."""
    D = gt.plane_x
    n, e_u, e_v, O = plane_basis(gt)
    rects = window_rects_uv(gt, e_u, e_v, O)
    psis = np.linspace(0, 2 * np.pi, npsi, endpoint=False)
    frac_win = []
    for phc in PHI:
        r = D * np.tan(np.radians(phc))
        y = r * np.cos(psis); z = r * np.sin(psis)
        # in-plane coords relative to O (frontal: e_u~+y, e_v~+z)
        qu = (np.column_stack([np.full_like(y, D), y, z]) - O) @ e_u
        qv = (np.column_stack([np.full_like(y, D), y, z]) - O) @ e_v
        inw = np.zeros(nsamp := len(psis), bool)
        for (u0, u1, v0, v1) in rects:
            inw |= (qu >= u0) & (qu <= u1) & (qv >= v0) & (qv <= v1)
        frac_win.append(inw.mean())
    return np.array(frac_win)


def per_sr(ds, which, stride):
    """Aggregate window & opaque returns per steradian vs phi, over a scan subset."""
    bagt, gtt, allns, subsets = CFG[ds]
    ns = subsets.get(which, allns)
    win = np.zeros(len(PHI)); opa = np.zeros(len(PHI))
    fw = np.zeros(len(PHI)); nfr = 0
    Rmed = []
    for nidx in ns:
        bag, gtp = bagt.format(n=nidx), gtt.format(n=nidx)
        import os
        if not os.path.isdir(bag):
            continue
        frames = read_all(bag)[::stride]
        gt = load_gt(gtp)
        s = segment(frames, gt, 24.0, 1)
        R = np.maximum(s.R, 1e-9)
        phi = np.degrees(np.arccos(np.clip(s.ned[:, 0] / R, -1, 1)))
        isw = s.cls == GLASS
        isop = (~isw) & (np.abs(s.behind_gt) < 0.12) & (s.ned[:, 0] > 0)
        wi = np.clip(((phi - 0.0) / 5.0).astype(int), 0, len(PHI) - 1)
        for b in range(len(PHI)):
            win[b] += isw[wi == b].sum()
            opa[b] += isop[wi == b].sum()
        fw += gt_window_fraction(gt) * len(frames)
        nfr += len(frames)
        Rmed.append(np.median(R[isw]))
    fw /= max(nfr, 1)                                   # avg window fraction per ring
    ring_sr = 2 * np.pi * np.sin(np.radians(PHI)) * DPHI
    win_sr = win / (nfr * ring_sr * np.clip(fw, 1e-3, 1))
    opa_sr = opa / (nfr * ring_sr * np.clip(1 - fw, 1e-3, 1))
    return win_sr, opa_sr, (np.median(Rmed) if Rmed else np.nan)


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    for j, ds in enumerate(("l6", "oxy", "bech")):
        for which, col in (("near", "tab:blue"), ("far", "tab:red")):
            try:
                wsr, osr, Rm = per_sr(ds, which, stride)
            except Exception as e:
                print(f"{ds}/{which}: {e}"); continue
            P = wsr / np.where(osr > 0, osr, np.nan)
            ax[j].plot(PHI, osr / np.nanmax(osr), "--", color=col, alpha=0.5,
                       label=f"{which} beam pattern n(phi) [norm]")
            ax[j].plot(PHI, P, "-o", color=col, label=f"{which} P_return (R~{Rm:.2f}m)")
        ax[j].axhline(1, color="k", lw=0.6, ls=":")
        ax[j].set_title(f"{ds}: through-window return prob vs incidence\n"
                        "(dashed = beam pattern that was divided out)")
        ax[j].set_xlabel("incidence = angle from boresight (deg)")
        ax[j].set_ylabel("P_return  (window / opaque)"); ax[j].set_ylim(0, 1.6)
        ax[j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("analysis/return_prob_corrected.png", dpi=110)
    print("wrote analysis/return_prob_corrected.png")

    # ---- noise floor: faint-edge intensity vs range ----
    print("\n=== noise-floor estimate: faintest DETECTED intensity vs range (I_min ~ (T/C) R^2) ===")
    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))
    for ds, c in (("bech", "crimson"), ("oxy", "dodgerblue"), ("l6", "seagreen")):
        G, _, _ = collect(ds, which="all", stride=stride)
        R = G["R"]; I = G["I"]
        redges = np.linspace(np.percentile(R, 2), np.percentile(R, 98), 14)
        rc, imin = [], []
        for a, b in zip(redges[:-1], redges[1:]):
            m = (R >= a) & (R < b)
            if m.sum() > 200:
                rc.append(0.5 * (a + b)); imin.append(np.percentile(I[m], 3))
        rc, imin = np.array(rc), np.array(imin)
        ax2[0].plot(rc, imin, "-o", color=c, label=f"{ds}")
        if len(rc) > 3:
            k = np.sum(imin * rc**2) / np.sum(rc**4)        # fit I_min = k R^2
            ax2[0].plot(rc, k * rc**2, ":", color=c)
            print(f"  {ds}: faint-edge I_min(3pct) fit ~ {k:.1f} * R^2  "
                  f"(=> T/C ~ {k:.1f} counts/m^2-equiv); I_min at 1.5m ~ {k*1.5**2:.0f}")
    ax2[0].set_title("faintest detected intensity vs range (3rd pct)\n(rising ~R^2 = noise floor)")
    ax2[0].set_xlabel("range R (m)"); ax2[0].set_ylabel("I_min (counts)"); ax2[0].legend()
    # return count vs range for the dim glass (Bech) -- dropout
    for ds, c in (("bech", "crimson"), ("oxy", "dodgerblue")):
        rows = []
        bagt, gtt, allns, subs = CFG[ds]
        import os
        for nidx in allns:
            bag, gtp = bagt.format(n=nidx), gtt.format(n=nidx)
            if not os.path.isdir(bag):
                continue
            fr = read_all(bag)[::stride]; gt = load_gt(gtp)
            s = segment(fr, gt, 24.0, 1)
            isw = s.cls == GLASS
            if isw.sum() < 30:
                continue
            rows.append((np.median(s.R[isw]), isw.sum() / len(fr)))
        rows = np.array(rows)
        ax2[1].scatter(rows[:, 0], rows[:, 1], color=c, label=ds, alpha=0.7)
    ax2[1].set_title("through-window returns per frame vs range\n(dim glass drops out with range)")
    ax2[1].set_xlabel("range R (m)"); ax2[1].set_ylabel("window returns / frame"); ax2[1].legend()
    fig2.tight_layout(); fig2.savefig("analysis/noise_floor.png", dpi=110)
    print("wrote analysis/noise_floor.png")


if __name__ == "__main__":
    main()
