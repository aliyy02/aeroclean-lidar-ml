"""A plain-language schematic of the intensity-vs-angle model (no jargon, no ROS).

Left: the three ways a surface can send the laser back (the whole model).
Right: the three glasses as those ingredients added up, matching the real curves.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

th = np.linspace(0, 75, 200)

# the three "ingredients" (schematic)
floor = 0.0 * th + 1.0                                  # matte: same at every angle
spike = np.exp(-(th / 6.0) ** 2)                        # mirror: only when head-on
graze = (th / 75.0) ** 3.0                              # glass edge-on: rises near grazing

# the three glasses = different amounts of each ingredient (schematic, ~matches data)
L6 = 85 + 170 * spike + 18 * graze
Oxy = 130 + 33 * np.exp(-(th / 9.0) ** 2) + 40 * (th / 75.0) ** 2.5
Bech = np.clip(65 + 55 * np.exp(-(th / 6.0) ** 2) + 200 * (th / 72.0) ** 3.5, 0, 255)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# ---- left: the three ingredients ----
a = ax[0]
a.plot(th, 90 * floor, lw=3, color="tab:green", label="1) MATTE (wall/chalk): same at every angle")
a.plot(th, 250 * spike, lw=3, color="tab:blue", label="2) MIRROR (shiny): only when head-on")
a.plot(th, 250 * graze, lw=3, color="tab:red", label="3) GLASS edge-on: flares near grazing")
a.set_title("The 3 ways a surface sends the laser back\n(the model just adds these up)", fontsize=12)
a.set_xlabel("how slanted the beam hits  (0 = straight on  →  75 = nearly edge-on)")
a.set_ylabel("light sent back toward the LiDAR")
a.legend(loc="upper center", fontsize=10); a.set_ylim(0, 270)
a.annotate("a 'flash' straight on", (2, 250), (18, 235), fontsize=9,
           arrowprops=dict(arrowstyle="->"))
a.annotate("steady at all angles", (50, 90), (40, 120), fontsize=9,
           arrowprops=dict(arrowstyle="->"))
a.annotate("only near edge-on", (70, 200), (45, 200), fontsize=9,
           arrowprops=dict(arrowstyle="->"))

# ---- right: the three glasses ----
b = ax[1]
b.plot(th, L6, lw=3, color="seagreen", label="L6  = mirror flash + steady glow")
b.plot(th, Oxy, lw=3, color="dodgerblue", label="Oxy = mostly steady glow (flat)")
b.plot(th, Bech, lw=3, color="crimson", label="Bech = dark, only flares edge-on")
b.set_title("The three real glasses = different mixes", fontsize=12)
b.set_xlabel("how slanted the beam hits  (0 = straight on  →  75 = nearly edge-on)")
b.set_ylabel("brightness that comes back (0-255)")
b.set_ylim(0, 270); b.legend(loc="upper center", fontsize=10)
b.annotate("bright flash\nhead-on", (1, 255), (12, 215), fontsize=9, color="seagreen",
           arrowprops=dict(arrowstyle="->", color="seagreen"))
b.annotate("dark head-on\n(tint absorbs the laser)", (8, 66), (20, 95), fontsize=9,
           color="crimson", arrowprops=dict(arrowstyle="->", color="crimson"))
b.annotate("flares up edge-on\n(like a lake seen low)", (72, 250), (40, 235),
           fontsize=9, color="crimson", arrowprops=dict(arrowstyle="->", color="crimson"))

fig.tight_layout()
fig.savefig("analysis/model_explained.png", dpi=120)
print("wrote analysis/model_explained.png")
