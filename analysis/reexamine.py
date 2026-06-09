"""Re-test two user notes against the data:
  (B) Bech also has a HEAD-ON FLASH (a near-normal bump) on top of the grazing rise.
  (C) Oxy windows had a CURTAIN behind the glass -> the dense flat returns may be the
      curtain (matte fabric), not the glass surface. If so, in-window returns split into
      a sparse glass FRONT (~0, Fresnel: dim at normal) + a dense curtain BACK (~0.29 m, flat).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from analysis.agg import collect, binned_median           # noqa: E402

FINE = np.array([0, 2, 4, 6, 9, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], float)


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    Go, Fo, _ = collect("oxy", which="all", stride=stride)
    Gb, Fb, _ = collect("bech", which="all", stride=stride)
    Gl, Fl, _ = collect("l6", which="all", stride=stride)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))

    # --- C: Oxy depth two-layer test ---
    bf = Go["bf"]
    bins = np.linspace(-0.15, 0.7, 120)
    ax[0].hist(bf, bins=bins, color="dodgerblue", alpha=0.7)
    ax[0].axvspan(-0.12, 0.12, color="green", alpha=0.15, label="FRONT (glass surface?)")
    ax[0].axvspan(0.18, 0.45, color="orange", alpha=0.15, label="BACK (curtain?)")
    ax[0].set_title("Oxy in-window depth behind front\n(two layers = glass + curtain?)")
    ax[0].set_xlabel("behind fitted front (m)"); ax[0].legend()
    front = (bf < 0.12); back = (bf > 0.18) & (bf < 0.45); deep = bf > 0.45
    print(f"OXY in-window depth split: FRONT(<0.12m)={front.mean()*100:.0f}%  "
          f"BACK(0.18-0.45)={back.mean()*100:.0f}%  DEEP(>0.45)={deep.mean()*100:.0f}%")
    print(f"  FRONT intensity median={np.median(Go['I'][front]):.0f}  "
          f"BACK intensity median={np.median(Go['I'][back]):.0f}")

    # --- C: Oxy front-layer vs back-layer intensity vs incidence ---
    for mask, c, lab in [(front, "green", "FRONT layer (glass surface)"),
                         (back, "orange", "BACK layer (curtain)")]:
        ctr, med, q1, q3, n = binned_median(Go["th"][mask], Go["I"][mask], FINE)
        ax[1].plot(ctr, med, "-o", color=c, label=lab)
        ax[1].fill_between(ctr, q1, q3, color=c, alpha=0.15)
    ax[1].set_title("Oxy: FRONT vs BACK intensity vs incidence\n"
                    "(glass=dim-at-normal/Fresnel ; curtain=flat matte)")
    ax[1].set_xlabel("incidence (deg)"); ax[1].set_ylabel("intensity")
    ax[1].set_ylim(0, 256); ax[1].legend(fontsize=8)

    # --- B: near-normal flash, Bech vs Oxy vs L6 ---
    NN = np.array([0, 1.5, 3, 5, 8, 12, 18, 26, 36, 48, 62, 75], float)
    for G, c, lab in [(Gb, "crimson", "Bech"), (Go, "dodgerblue", "Oxy"), (Gl, "seagreen", "L6")]:
        ctr, med, q1, q3, n = binned_median(G["th"], G["I"], NN)
        ax[2].plot(ctr, med, "-o", color=c, label=lab)
    ax[2].set_title("Near-normal detail: is there a HEAD-ON FLASH?\n"
                    "(bump at ~0 above the floor)")
    ax[2].set_xlabel("incidence (deg)"); ax[2].set_ylabel("intensity")
    ax[2].set_ylim(0, 256); ax[2].legend()
    # quantify Bech flash: value at 0-2 vs the floor (8-12)
    for G, name in [(Gb, "Bech"), (Go, "Oxy"), (Gl, "L6")]:
        m0 = G["th"] < 2; mfl = (G["th"] > 8) & (G["th"] < 14)
        print(f"{name}: I(0-2deg)={np.median(G['I'][m0]):.0f}  I(8-14deg floor)="
              f"{np.median(G['I'][mfl]):.0f}  flash bump={np.median(G['I'][m0])-np.median(G['I'][mfl]):+.0f}")

    fig.tight_layout(); fig.savefig("analysis/reexamine.png", dpi=110)
    print("wrote analysis/reexamine.png")


if __name__ == "__main__":
    main()
