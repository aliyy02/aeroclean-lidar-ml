"""Turn a real L2 bag + its test_bed GT .txt into per-frame labeled point clouds.

  bag (native L2) + gt.txt  ->  for each frame:  native -> corrected NED (remap + de-roll)
                                                 -> label {glass, other, ground, interior}
                                                 -> drop everything off the panel/floor
                                                 -> .npz {xyz, intensity, label, normal}

The floor plane is found once from the aggregate (the pose is static within a bag). `alpha`
(the boresight roll) is a config knob; pass `--qa` to render an overlay for tuning it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np

from .gt_parse import GroundTruth, load_gt
from .frames import to_corrected
from .label import label_points, detect_floor, plane_basis, estimate_panel_offset, \
    LabelParams, DROP, GLASS, NOT_GLASS, GROUND, INTERIOR
from .io_bag import Frame, read_all

LABEL_NAMES = {NOT_GLASS: "not_glass", GLASS: "glass", GROUND: "ground",
               INTERIOR: "interior"}


@dataclass
class LabeledFrame:
    stamp_ns: int
    xyz: np.ndarray          # (N,3) corrected NED, meters (kept points only)
    intensity: np.ndarray    # (N,)
    label: np.ndarray        # (N,) int
    normal: np.ndarray       # (N,3)


def process_frames(frames: List[Frame], gt: GroundTruth, alpha: float,
                   params: LabelParams = LabelParams(),
                   floor: Optional[Tuple[np.ndarray, float]] = None
                   ) -> List[LabeledFrame]:
    """Label every frame; drop cropped (off-panel) points. Floor and the panel-depth
    offset (the accumulated measurement recession of the real panel behind the GT plane)
    are estimated once from the aggregate (the pose is static within a bag)."""
    corrected = [to_corrected(f.xyz, alpha) for f in frames]
    agg = np.vstack(corrected) if corrected else np.zeros((0, 3))
    if floor is None and corrected:
        floor = detect_floor(agg, gt)
    delta = estimate_panel_offset(agg, gt) if corrected else 0.0
    if delta > 0:
        params = replace(params, interior_cut=delta + params.interior_margin)
    out: List[LabeledFrame] = []
    for f, c in zip(frames, corrected):
        res = label_points(c, gt, params, floor=floor)
        keep = res.labels != DROP
        out.append(LabeledFrame(stamp_ns=f.stamp_ns, xyz=c[keep],
                                intensity=f.intensity[keep], label=res.labels[keep],
                                normal=res.normals[keep]))
    return out


def write_outputs(labeled: List[LabeledFrame], gt: GroundTruth, alpha: float,
                  params: LabelParams, out_dir: str) -> None:
    """Write per-frame .npz + meta.json + index.csv."""
    os.makedirs(out_dir, exist_ok=True)
    index_rows = []
    for i, lf in enumerate(labeled):
        fn = f"frame_{i:04d}.npz"
        np.savez_compressed(os.path.join(out_dir, fn),
                            xyz=lf.xyz.astype(np.float32),
                            intensity=lf.intensity.astype(np.float32),
                            label=lf.label.astype(np.int16),
                            normal=lf.normal.astype(np.float32))
        counts = {name: int((lf.label == lid).sum()) for lid, name in LABEL_NAMES.items()}
        index_rows.append({"file": fn, "stamp_ns": lf.stamp_ns,
                           "n": lf.xyz.shape[0], **counts})

    meta = {
        "alpha_deg": alpha,
        "axis_map": "ned=(sz,sx,sy)",
        "plane_x": gt.plane_x,
        "plane_normal": gt.plane_normal.tolist(),
        "pose": gt.pose,
        "lidar_bed_m": gt.lidar_bed.tolist(),
        "label_ids": {v: k for k, v in LABEL_NAMES.items()},
        "params": vars(params),
        "n_frames": len(labeled),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    with open(os.path.join(out_dir, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index_rows[0]))
        w.writeheader()
        w.writerows(index_rows)


def qa_overlay(frames: List[Frame], gt: GroundTruth, alpha: float, out_path: str,
               params: LabelParams = LabelParams()) -> None:
    """Render the aggregate labeled cloud (head-on) with GT windows, to tune alpha."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    labeled = process_frames(frames, gt, alpha, params)
    xyz = np.vstack([lf.xyz for lf in labeled])
    lab = np.concatenate([lf.label for lf in labeled])
    # Head-on view in the facade's in-plane (u,v) coords. Facade points are shown at their
    # beam->plane intersection (back-projected) so the see-through spread collapses and the
    # windows are crisp; ground/other shown at projected actual position.
    n, e_u, e_v, O = plane_basis(gt)
    d = gt.plane_d
    nP = xyz @ n
    ahead = nP < -1e-9
    t = np.where(ahead, -d / np.where(ahead, nP, 1.0), 0.0)
    P = xyz.copy()
    facade = np.isin(lab, [GLASS, INTERIOR, NOT_GLASS]) & ahead
    P[facade] = xyz[facade] * t[facade, None]
    u = (P - O) @ e_u
    v = (P - O) @ e_v
    colors = {NOT_GLASS: "gold", GLASS: "dodgerblue", GROUND: "limegreen",
              INTERIOR: "red"}
    fig, ax = plt.subplots(figsize=(11, 8))
    for lid, name in LABEL_NAMES.items():
        m = lab == lid
        if m.any():
            ax.scatter(u[m], -v[m], s=2, c=colors[lid],
                       label=f"{name} ({m.sum()})", alpha=0.5)
    for w in gt.windows.values():
        us = [(c - O) @ e_u for c in w.values()]
        vs = [(c - O) @ e_v for c in w.values()]
        ax.add_patch(Rectangle((min(us), -max(vs)), max(us) - min(us),
                               max(vs) - min(vs), fill=False, ec="red", lw=1.5))
    ax.set_aspect("equal"); ax.legend(markerscale=4, loc="upper right")
    ax.set_xlabel("in-plane u (m)"); ax.set_ylabel("in-plane v, up (m)")
    ax.set_title(f"QA overlay  alpha={alpha:+.1f} deg  (red = GT windows)")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def label_scan(bag_path: str, gt_path: str, out_dir: str, alpha: float = 24.0,
               params: LabelParams = LabelParams(), qa: bool = False) -> None:
    """Full pipeline for one bag (reads ROS bag, writes labeled .npz)."""
    gt = load_gt(gt_path)
    frames = read_all(bag_path)
    labeled = process_frames(frames, gt, alpha, params)
    write_outputs(labeled, gt, alpha, params, out_dir)
    if qa:
        qa_overlay(frames, gt, alpha, os.path.join(out_dir, "qa_overlay.png"), params)
    kept = sum(lf.xyz.shape[0] for lf in labeled)
    raw = sum(f.xyz.shape[0] for f in frames)
    print(f"{len(frames)} frames | kept {kept}/{raw} points -> {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Label a real L2 scan against test_bed GT.")
    ap.add_argument("bag", help="path to the rosbag2 mcap directory")
    ap.add_argument("gt", help="path to the test_bed ground-truth .txt")
    ap.add_argument("out", help="output directory")
    ap.add_argument("--alpha", type=float, default=24.0, help="boresight roll deg")
    ap.add_argument("--tau-front", type=float, default=0.05)
    ap.add_argument("--tau-panel", type=float, default=0.05)
    ap.add_argument("--grid-margin", type=float, default=0.05)
    ap.add_argument("--qa", action="store_true", help="also write qa_overlay.png")
    a = ap.parse_args(argv)
    params = LabelParams(tau_front=a.tau_front, tau_panel=a.tau_panel,
                         grid_margin=a.grid_margin)
    label_scan(a.bag, a.gt, a.out, alpha=a.alpha, params=params, qa=a.qa)


if __name__ == "__main__":
    main()
