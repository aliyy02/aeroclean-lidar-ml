#!/usr/bin/env python3
"""Offline matplotlib gallery of generated buildings -- a fast, honest preview of the
geometry the spawner sends to UE5 (same Box list, same world NED, same class labels).

No sim needed. Backface-culls + painter-sorts the box faces and draws them in 2D under an
oblique orthographic camera, coloured by the 8-class material id. Use it to eyeball facade
diversity (S1..S6) without waiting on UE5.

    python3 -m buildings.render_gallery            # writes /tmp/building_gallery.png
"""
from __future__ import annotations

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from dataclasses import replace

from forward_model import materials as M
from buildings.build import sample_building_params, build_building
from buildings.footprint import footprint_from_vertices

# class -> rgb (0..1), matches buildings/ros_publish.py colours
_COLORS = {0: (.59, .59, .59), 1: (.86, .16, .16), 2: (.16, .78, .90), 3: (.16, .35, .90),
           4: (.67, .27, .90), 5: (.86, .51, .16), 6: (.16, .70, .24), 7: (.92, .92, .92)}

# the 6 box faces as (axis, sign) and the 4 local-corner sign patterns for each
_FACE_LOCAL = {
    (0, +1): [(+1, -1, -1), (+1, +1, -1), (+1, +1, +1), (+1, -1, +1)],
    (0, -1): [(-1, -1, -1), (-1, +1, -1), (-1, +1, +1), (-1, -1, +1)],
    (1, +1): [(-1, +1, -1), (+1, +1, -1), (+1, +1, +1), (-1, +1, +1)],
    (1, -1): [(-1, -1, -1), (+1, -1, -1), (+1, -1, +1), (-1, -1, +1)],
    (2, +1): [(-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1)],
    (2, -1): [(-1, -1, -1), (+1, -1, -1), (+1, +1, -1), (-1, +1, -1)],
}


def _camera(view_dir):
    f = np.asarray(view_dir, float); f /= np.linalg.norm(f)
    up_w = np.array([0.0, 0.0, -1.0])                 # NED up
    right = np.cross(f, up_w); right /= np.linalg.norm(right)
    cam_up = np.cross(right, f)
    return f, right, cam_up


def _box_faces(b, f, right, cam_up):
    """Return (polygon2d, color, depth) for each camera-facing face of a box."""
    out = []
    for (axis, sign), corners in _FACE_LOCAL.items():
        nrm = sign * b.R[:, axis]
        if nrm @ f >= 0:                              # backface cull: keep faces toward camera
            continue
        pts = np.array([b.center + b.R @ (np.array(c) * b.half) for c in corners])
        poly = np.column_stack([pts @ right, pts @ cam_up])
        out.append((poly, _COLORS[b.cls], float(pts.mean(0) @ f)))
    return out


def render(ax, bld, view_dir=(1.0, 0.7, 0.35), title=""):
    f, right, cam_up = _camera(view_dir)
    faces = []
    for b in bld.boxes:
        faces.extend(_box_faces(b, f, right, cam_up))
    faces.sort(key=lambda t: -t[2])                   # painter: farthest first
    polys = [p for p, _, _ in faces]
    cols = [c for _, c, _ in faces]
    pc = PolyCollection(polys, facecolors=cols, edgecolors=(0, 0, 0, 0.25), linewidths=0.15)
    ax.add_collection(pc)
    allpts = np.vstack(polys)
    ax.set_xlim(allpts[:, 0].min() - 1, allpts[:, 0].max() + 1)
    ax.set_ylim(allpts[:, 1].min() - 1, allpts[:, 1].max() + 1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=10)


def _building_with_system(system, region, n_floors, seed):
    """Sample params until we land on the wanted system, then pin the floor count."""
    rng = np.random.default_rng(seed)
    for _ in range(400):
        p = sample_building_params(rng, region=region)
        if p.system == system:
            p = replace(p, n_floors=n_floors)
            fp = footprint_from_vertices([(-8, -7), (8, -7), (8, 7), (-8, 7)], kind="square")
            return build_building(fp, p, np.random.default_rng(seed))
    raise RuntimeError(f"no {system} in {region} after 400 draws")


def render_random_batch(out, n=8, seed=0):
    """Render n buildings drawn exactly as a real capture batch would draw them
    (mixed region, mixed system, varied floors/footprint) -- the honest output distribution."""
    from buildings.build import sample_building
    cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows))
    rng = np.random.default_rng(seed)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off"); continue
        region = "gcc" if rng.random() < 0.5 else "lebanon"
        bld = sample_building(rng, region=region, square_prob=1.0)
        p = bld.params
        render(ax, bld, title=f"{region}  {p.system}  {p.n_floors}fl")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                          markerfacecolor=_COLORS[k], markeredgecolor="k",
                          label=M.CLASS_NAMES[k]) for k in sorted(_COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=9, frameon=False)
    fig.suptitle("Random capture batch - what the generator actually produces (offline preview)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out, dpi=110)
    print("wrote", out)


def main():
    if "--random" in sys.argv:
        seed = 0
        if "--seed" in sys.argv:
            seed = int(sys.argv[sys.argv.index("--seed") + 1])
        render_random_batch("/tmp/building_batch.png", n=8, seed=seed)
        return
    panels = [
        ("S1", "gcc", 7, "S1  unitized curtain wall (captured grid) - GCC"),
        ("S2", "gcc", 7, "S2  SSG / flush glazing (thin seams) - GCC"),
        ("S3", "lebanon", 6, "S3  stick-built (deep expressed mullions)"),
        ("S5", "gcc", 7, "S5  ribbon / strip windows"),
        ("S6", "lebanon", 6, "S6  punched windows in wall - Lebanon"),
        ("S6", "gcc", 7, "S6  punched windows in wall - GCC"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (sysid, region, nfl, title) in zip(axes.flat, panels):
        bld = _building_with_system(sysid, region, nfl, seed=7)
        render(ax, bld, title=title)
    # legend
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                          markerfacecolor=_COLORS[k], markeredgecolor="k",
                          label=M.CLASS_NAMES[k]) for k in sorted(_COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=9, frameon=False)
    fig.suptitle("Building generator - facade system diversity (offline preview, world NED)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/building_gallery.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
