"""Parse a `test_bed_multiwindow.py` ground-truth .txt.

The file gives, per window, 4 corners (UL,UR,LL,LR) in the test-bed 'LiDAR frame'
(== the mounting-hole / assumed body-NED frame), in **mm**; a facade plane; the rig
pose; and the lidar position in the bed frame. This parser returns everything in
**meters** so it lines up with the (meters) point cloud.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

MM_TO_M = 1e-3

_CELL_RE = re.compile(r"^\s*(r\d+c\d+):\s*$")
_CORNER_RE = re.compile(
    r"^\s*(UL|UR|LL|LR):\s*\[\s*([-\d.eE]+),\s*([-\d.eE]+),\s*([-\d.eE]+)\s*\]")
_POSE_RE = re.compile(
    r"Pose:\s*X=([-\d.eE]+)\s*Y=([-\d.eE]+)\s*Z=([-\d.eE]+)\s*\|\s*"
    r"Yaw=([-\d.eE]+)\s*Pitch=([-\d.eE]+)\s*Roll=([-\d.eE]+)")
_LIDAR_RE = re.compile(
    r"LiDAR \(bed frame\):\s*\[\s*([-\d.eE]+),\s*([-\d.eE]+),\s*([-\d.eE]+)\s*\]")
_PLANE_RE = re.compile(
    r"Facade plane:\s*([-\d.eE]+)x\s*\+\s*([-\d.eE]+)y\s*\+\s*([-\d.eE]+)z\s*\+\s*"
    r"([-\d.eE]+)\s*=\s*0")


@dataclass
class GroundTruth:
    windows: Dict[str, Dict[str, np.ndarray]]   # cell -> corner -> (3,) meters
    plane_normal: np.ndarray                     # (3,) unit, in assumed body-NED, toward sensor
    plane_d: float                               # facade plane: plane_normal . X + plane_d = 0
    pose: Dict[str, float]                       # X,Y,Z (mm-ruler), yaw,pitch,roll (deg)
    lidar_bed: np.ndarray                        # (3,) meters, lidar pos in bed frame

    @property
    def plane_x(self) -> float:
        """Facade depth along +x (valid for an x-facing facade; pose yaw=pitch=0)."""
        return -self.plane_d / self.plane_normal[0]

    def grid_bbox(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Raw 3D bbox of all window corners in (y, z), meters. Used for coarse gating;
        for labeling use the in-plane (u,v) box (the facade may be tilted)."""
        ys = [c[1] for w in self.windows.values() for c in w.values()]
        zs = [c[2] for w in self.windows.values() for c in w.values()]
        return (min(ys), max(ys)), (min(zs), max(zs))


def parse_gt(text: str) -> GroundTruth:
    """Parse the ground-truth text (meters out)."""
    windows: Dict[str, Dict[str, np.ndarray]] = {}
    cell = None
    for line in text.splitlines():
        mc = _CELL_RE.match(line)
        if mc:
            cell = mc.group(1)
            windows[cell] = {}
            continue
        mp = _CORNER_RE.match(line)
        if mp and cell is not None:
            xyz = np.array([float(mp.group(i)) for i in (2, 3, 4)]) * MM_TO_M
            windows[cell][mp.group(1)] = xyz

    pose_m = _POSE_RE.search(text)
    if pose_m is None:
        raise ValueError("could not find a Pose line in GT text")
    pose = dict(zip(("X", "Y", "Z", "yaw", "pitch", "roll"),
                    (float(pose_m.group(i)) for i in range(1, 7))))

    lid_m = _LIDAR_RE.search(text)
    lidar_bed = (np.array([float(lid_m.group(i)) for i in (1, 2, 3)]) * MM_TO_M
                 if lid_m else np.full(3, np.nan))

    pl = _PLANE_RE.search(text)
    if pl is None:
        raise ValueError("could not find a Facade plane line in GT text")
    normal = np.array([float(pl.group(i)) for i in (1, 2, 3)])
    d = float(pl.group(4)) * MM_TO_M
    nrm = np.linalg.norm(normal)
    normal = normal / nrm
    d = d / nrm
    # plane: normal·X + d = 0. normal points toward the sensor (origin has normal·0+d=d>0).

    if not windows:
        raise ValueError("no windows parsed from GT text")
    return GroundTruth(windows=windows, plane_normal=normal, plane_d=d,
                       pose=pose, lidar_bed=lidar_bed)


def load_gt(path: str) -> GroundTruth:
    """Read a GT .txt file and parse it."""
    with open(path) as f:
        return parse_gt(f.read())
