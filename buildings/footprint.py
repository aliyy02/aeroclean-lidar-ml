"""Building footprints: closed polygons in the world X-Y plane + per-face outward normals.

World is NED (X=North, Y=East, Z=Down; up = -Z). A footprint's edges become vertical
wall faces; each face's outward normal is horizontal (nz = 0) and points away from the
polygon interior. Squares give axis-aligned (±X/±Y) normals -- no rotation-convention
risk -- which is why we build those first; chamfered polygons add angled faces.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Face:
    start: tuple        # (x, y) edge start in world
    end: tuple          # (x, y) edge end
    normal: tuple       # (nx, ny, 0) unit outward normal
    tangent: tuple      # (tx, ty, 0) unit along the edge
    width: float        # edge length (m)


@dataclass
class Footprint:
    vertices: list      # [(x, y), ...] in order
    faces: list         # [Face, ...]
    kind: str           # "square" | "polygon"


def footprint_from_vertices(vertices, kind: str = "square") -> Footprint:
    """Build a Footprint (faces + outward normals) from an ordered vertex list.

    Outward normals are chosen by **winding order**, not a centroid test: for a CCW polygon the
    interior lies to the left of each edge tangent, so the outward normal is the right-hand
    perpendicular (t_y, -t_x). This is correct for concave footprints too (L/U/T notches), where
    a centroid heuristic flips the normal on inner edges.
    """
    verts = [tuple(map(float, v)) for v in vertices]
    n = len(verts)
    area2 = sum(verts[i][0] * verts[(i + 1) % n][1] - verts[(i + 1) % n][0] * verts[i][1]
                for i in range(n))
    ccw = area2 > 0
    faces = []
    for i in range(n):
        a = np.array(verts[i]); b = np.array(verts[(i + 1) % n])
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        tan = edge / length
        perp = np.array([tan[1], -tan[0]]) if ccw else np.array([-tan[1], tan[0]])
        faces.append(Face(
            start=tuple(a), end=tuple(b),
            normal=(float(perp[0]), float(perp[1]), 0.0),
            tangent=(float(tan[0]), float(tan[1]), 0.0),
            width=length,
        ))
    return Footprint(vertices=verts, faces=faces, kind=kind)


def _rectangle(rng) -> list:
    D = float(rng.uniform(10.0, 26.0))      # X (North) extent
    W = float(rng.uniform(10.0, 26.0))      # Y (East) extent
    return [(-D / 2, -W / 2), (D / 2, -W / 2), (D / 2, W / 2), (-D / 2, W / 2)]


def _chamfer_corners(verts, rng) -> list:
    """Cut 1-3 corners off a polygon, replacing each with an angled edge."""
    n = len(verts)
    k = int(rng.integers(1, min(3, n - 1) + 1))
    corners = sorted(rng.choice(n, size=k, replace=False), reverse=True)
    out = [np.array(v, dtype=float) for v in verts]
    for ci in corners:
        v = out[ci]
        p = out[(ci - 1) % len(out)]
        q = out[(ci + 1) % len(out)]
        cp = float(rng.uniform(1.5, 4.0))
        a = v + min(cp, 0.4 * np.linalg.norm(p - v)) * (p - v) / np.linalg.norm(p - v)
        b = v + min(cp, 0.4 * np.linalg.norm(q - v)) * (q - v) / np.linalg.norm(q - v)
        out[ci:ci + 1] = [a, b]
    return [tuple(v) for v in out]


def _l_shape(rng) -> list:
    """Rectangle with a rectangular notch cut from the top-right corner (an L). Axis-aligned, CCW."""
    D = float(rng.uniform(16, 26)); W = float(rng.uniform(16, 26))
    cd = float(rng.uniform(5, 0.55 * D)); cw = float(rng.uniform(5, 0.55 * W))
    return [(-D / 2, -W / 2), (D / 2, -W / 2), (D / 2, W / 2 - cw),
            (D / 2 - cd, W / 2 - cw), (D / 2 - cd, W / 2), (-D / 2, W / 2)]


def _u_shape(rng) -> list:
    """Rectangle with a notch cut from the middle of the +Y edge (a U). Axis-aligned, CCW."""
    D = float(rng.uniform(16, 26)); W = float(rng.uniform(16, 26))
    nd = float(rng.uniform(4, 0.4 * D)); nw = float(rng.uniform(5, 0.6 * W))
    return [(-D / 2, -W / 2), (D / 2, -W / 2), (D / 2, W / 2), (nd / 2, W / 2),
            (nd / 2, W / 2 - nw), (-nd / 2, W / 2 - nw), (-nd / 2, W / 2), (-D / 2, W / 2)]


def _t_shape(rng) -> list:
    """A T: full-width top bar + a narrower stem at the bottom centre. Axis-aligned, CCW."""
    D = float(rng.uniform(18, 26)); W = float(rng.uniform(16, 26))
    sd = float(rng.uniform(6, 0.5 * D)); sw = float(rng.uniform(5, 0.55 * W))
    return [(-sd / 2, -W / 2), (sd / 2, -W / 2), (sd / 2, -W / 2 + sw), (D / 2, -W / 2 + sw),
            (D / 2, W / 2), (-D / 2, W / 2), (-D / 2, -W / 2 + sw), (-sd / 2, -W / 2 + sw)]


_RECTILINEAR = (_l_shape, _u_shape, _t_shape)


def sample_footprint(rng, square_prob: float = 0.6, allow_angled: bool = False) -> Footprint:
    """Sample a footprint.

    - prob `square_prob`: a rectangle (kind="square", axis-aligned).
    - else: an axis-aligned rectilinear L/U/T (kind="polygon") -- every face still faces +/-X or
      +/-Y, so it spawns correctly with the current identity-quaternion spawner (no rotated boxes).
    - `allow_angled=True` re-enables chamfered (angled-face) polygons; those need the rotated-box
      spawn convention and are off by default.
    """
    if rng.random() < square_prob:
        return footprint_from_vertices(_rectangle(rng), kind="square")
    if allow_angled and rng.random() < 0.5:
        return footprint_from_vertices(_chamfer_corners(_rectangle(rng), rng), kind="polygon")
    shape = _RECTILINEAR[int(rng.integers(len(_RECTILINEAR)))]
    return footprint_from_vertices(shape(rng), kind="polygon")
