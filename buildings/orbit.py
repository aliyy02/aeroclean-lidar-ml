"""Orbit capture poses: a position grid over each face + biased viewpoint sampling.

The sensor walks a grid over every face (1 m cells, 1.5 m for faces > 600 m^2). At each
cell it takes K shots, each with a standoff and orientation drawn from biased ranges:
standoff -> 1-2 m, pitch -> 0..-20 deg (looking slightly down), roll/yaw -> ~0. Positions
are exact world coordinates; the orientation is commanded and the true frame is recovered
per-capture by the rotation solve, so AirSim's attitude convention never matters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_UP = np.array([0.0, 0.0, -1.0])
# bias parameters (tunable)
STANDOFF_MEAN, STANDOFF_STD, STANDOFF_LO, STANDOFF_HI = 1.5, 0.6, 1.0, 4.0
PITCH_MEAN, PITCH_STD, PITCH_LIM = -10.0, 8.0, 40.0
ROLL_STD, ROLL_LIM = 6.0, 40.0
YAW_STD, YAW_LIM = 8.0, 30.0
LARGE_FACE_AREA = 600.0


@dataclass
class CapturePose:
    sensor_pos: np.ndarray      # (3,) world -- where we want the LIDAR
    outward: np.ndarray         # (3,) face outward normal (sensor sits along it)
    roll: float                 # degrees (deviation from facing the wall)
    pitch: float
    yaw_dev: float              # yaw deviation from the heading that faces the wall
    base_yaw: float             # heading toward the wall (deg)
    standoff: float
    face_idx: int

    @property
    def yaw(self) -> float:
        return self.base_yaw + self.yaw_dev


def grid_spacing(area: float) -> float:
    return 1.5 if area > LARGE_FACE_AREA else 1.0


def face_grid(face, total_h: float, spacing: float):
    """Grid of (face_point_world, outward_normal, up) over the face plane (margin-inset)."""
    start = np.array([face.start[0], face.start[1], 0.0])
    tan = np.array(face.tangent); n = np.array(face.normal)
    w = face.width
    ss = np.arange(spacing / 2, max(w - spacing / 2, spacing / 2) + 1e-9, spacing)
    hs = np.arange(spacing / 2, max(total_h - spacing / 2, spacing / 2) + 1e-9, spacing)
    out = []
    for s in ss:
        for h in hs:
            fp = start + s * tan + h * _UP        # up is -Z, so height h -> world z = -h
            out.append((fp, n, _UP))
    return out


def sample_pose(rng, face_point, outward_normal, face_idx: int = 0) -> CapturePose:
    n = np.asarray(outward_normal, float)
    standoff = float(np.clip(rng.normal(STANDOFF_MEAN, STANDOFF_STD), STANDOFF_LO, STANDOFF_HI))
    pitch = float(np.clip(rng.normal(PITCH_MEAN, PITCH_STD), -PITCH_LIM, PITCH_LIM))
    roll = float(np.clip(rng.normal(0.0, ROLL_STD), -ROLL_LIM, ROLL_LIM))
    yaw_dev = float(np.clip(rng.normal(0.0, YAW_STD), -YAW_LIM, YAW_LIM))
    base_yaw = math.degrees(math.atan2(-n[1], -n[0]))     # heading toward the wall
    return CapturePose(sensor_pos=np.asarray(face_point, float) + standoff * n, outward=n,
                       roll=roll, pitch=pitch, yaw_dev=yaw_dev, base_yaw=base_yaw,
                       standoff=standoff, face_idx=face_idx)


def capture_poses(building, rng, k: int = 4, budget: int | None = None):
    """Yield capture poses over the building. K shots per grid cell across every face; if `budget`
    is set and there are more candidate shots than that, draw a RANDOM `budget` of them spread over
    all faces (so a small per-building dataset still samples the whole building instead of just the
    first face -- the fix for the front-loading you get by truncating the stream)."""
    total_h = building.params.n_floors * building.params.floor_height
    shots = []                                            # (face_idx, face_pt, normal) per shot
    for fi, face in enumerate(building.footprint.faces):
        spacing = grid_spacing(face.width * total_h)
        for face_pt, n, _up in face_grid(face, total_h, spacing):
            shots.extend((fi, face_pt, n) for _ in range(k))
    if budget is not None and len(shots) > budget:
        keep = rng.choice(len(shots), size=budget, replace=False)
        shots = [shots[i] for i in keep]
    for fi, face_pt, n in shots:
        yield sample_pose(rng, face_pt, n, face_idx=fi)
