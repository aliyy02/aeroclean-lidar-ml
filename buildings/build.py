"""Extrude a footprint into a building of oriented boxes (world NED).

Each face gets a continuous, protruding aluminium mullion/transom grid plus per-cell
glass panes (sized by window width x height + sill), spandrel bands, and wall sections
(faces are mixed: some columns glazed, some solid). Every element is a `Box` with a
known world orientation R (columns = the face's [normal, tangent, up] axes), so a beam
landing on any of its 6 faces -- front or a protrusion reveal -- has an exact normal.

For square footprints R is axis-aligned (entries in {0, +/-1}); angled footprints rotate
the boxes into the face plane.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from forward_model import materials as M
from .footprint import sample_footprint, Footprint

_UP = np.array([0.0, 0.0, -1.0])          # world "up" in NED
_GLASS_THICK = 0.03
_GLASS_PREFIX = {M.GLASS_CLEAR: "GlassClear_", M.GLASS_COATED: "GlassCoated_", M.GLASS_LOWE: "GlassLowE_"}
_REGION_MIX = {                            # vision-glass class mix per region
    "gcc":     {M.GLASS_CLEAR: 0.10, M.GLASS_COATED: 0.55, M.GLASS_LOWE: 0.35},
    "lebanon": {M.GLASS_CLEAR: 0.50, M.GLASS_COATED: 0.35, M.GLASS_LOWE: 0.15},
}
# Facade SYSTEM mix per region + per-system facade ranges (see DIVERSITY_SPEC.md).
#   S1 unitized captured, S2 SSG/flush (thin seams), S3 stick (deep/expressed), S5 ribbon (banded)
_REGION_SYSTEM_MIX = {
    "gcc":     {"S1": 0.40, "S2": 0.28, "S3": 0.10, "S5": 0.12, "S6": 0.10},
    "lebanon": {"S6": 0.35, "S3": 0.20, "S5": 0.20, "S1": 0.15, "S2": 0.10},
}
_SYSTEMS = {   # mullion face(m), frame depth(m), glass recess(m), vision fraction, interm-transom p, wall-col p
    "S1": dict(mf=(0.050, 0.065), fd=(0.04, 0.06), rc=(0.04, 0.08), vis=(0.55, 0.85), it=0.6, wall=(0.00, 0.12)),
    "S2": dict(mf=(0.015, 0.025), fd=(0.02, 0.03), rc=(0.00, 0.02), vis=(0.60, 0.90), it=0.4, wall=(0.00, 0.10)),
    "S3": dict(mf=(0.060, 0.100), fd=(0.05, 0.12), rc=(0.05, 0.10), vis=(0.50, 0.70), it=0.7, wall=(0.05, 0.25)),
    "S5": dict(mf=(0.050, 0.080), fd=(0.04, 0.06), rc=(0.04, 0.08), vis=(0.30, 0.52), it=0.3, wall=(0.00, 0.10)),
}


@dataclass
class Box:
    name: str
    cls: int
    center: np.ndarray        # (3,) world
    R: np.ndarray             # (3,3) local->world, columns = [n, t, u]
    half: np.ndarray          # (3,) half extents along local [n, t, u]


@dataclass
class BuildingParams:
    n_floors: int
    floor_height: float
    module_width: float
    mullion_face: float
    frame_depth: float
    recess: float
    protrusion: float
    vision_fraction: float
    window_width_frac: float
    wall_col_prob: float
    full_wall_face_prob: float
    glass_class: int
    region: str
    system: str = "S1"                    # facade system type (see DIVERSITY_SPEC.md)
    intermediate_transom: bool = True     # a transom at the vision/spandrel boundary (diversity)
    punched: bool = False                 # S6: discrete windows in a solid wall (vs a glass grid)
    window_w: float = 1.4                 # S6 window width (m)
    window_h: float = 1.6                 # S6 window height (m)
    band_prob: float = 0.0                # prob a protruding horizontal band sits at each floor line
    band_depth: float = 0.0               # how far those bands stand proud of the facade (m)
    parapet: bool = False                 # a solid roof-edge upstand above the top floor
    plinth: bool = False                  # a protruding solid base band at the ground floor


@dataclass
class Building:
    boxes: list
    params: BuildingParams
    footprint: Footprint


def sample_building_params(rng, region: str = "gcc") -> BuildingParams:
    gmix = _REGION_MIX[region]
    gclass = int(rng.choice(list(gmix), p=list(gmix.values())))
    smix = _REGION_SYSTEM_MIX[region]
    system = str(rng.choice(list(smix), p=list(smix.values())))
    u = lambda ab: float(rng.uniform(*ab))
    # random protruding structures (string-course bands / parapet / plinth) -- ~1/3 of buildings
    # get floor-line bands; most get a parapet, many a plinth.
    bands = rng.random() < 0.35
    structs = dict(band_prob=u((0.2, 0.5)) if bands else 0.0,
                   band_depth=u((0.15, 0.40)) if bands else 0.0,
                   parapet=bool(rng.random() < 0.6), plinth=bool(rng.random() < 0.4))
    common = dict(n_floors=int(rng.integers(5, 13)), floor_height=float(rng.uniform(3.5, 4.5)),
                  protrusion=0.0, full_wall_face_prob=0.05,
                  glass_class=gclass, region=region, system=system, **structs)
    if system == "S6":                                     # punched windows in a solid wall
        return BuildingParams(module_width=u((2.5, 4.0)), mullion_face=u((0.05, 0.08)),
                              frame_depth=u((0.05, 0.10)), recess=u((0.05, 0.12)),
                              vision_fraction=0.0, window_width_frac=1.0, wall_col_prob=0.0,
                              intermediate_transom=False, punched=True,
                              window_w=u((1.0, 1.7)), window_h=u((1.2, 2.0)), **common)
    s = _SYSTEMS[system]
    # window size: wider bay range + sometimes a vision pane narrower than its bay (spandrel infill)
    wwf = 1.0 if rng.random() < 0.6 else u((0.6, 0.9))
    return BuildingParams(module_width=u((1.0, 2.4)), mullion_face=u(s["mf"]), frame_depth=u(s["fd"]),
                          recess=u(s["rc"]), vision_fraction=u(s["vis"]), window_width_frac=wwf,
                          wall_col_prob=u(s["wall"]),
                          intermediate_transom=bool(rng.random() < s["it"]), **common)


def _face_R(face) -> np.ndarray:
    n = np.array(face.normal); t = np.array(face.tangent)
    return np.column_stack([n, t, _UP])        # [n | t | u]


class _Namer:
    def __init__(self): self.k = {}
    def __call__(self, prefix):
        self.k[prefix] = self.k.get(prefix, 0) + 1
        return f"{prefix}{self.k[prefix]}"


def _box(name, cls, S2d, n2d, t2d, s_c, h_c, out_c, scale_ntu, R):
    xy = S2d + s_c * t2d + out_c * n2d
    center = np.array([xy[0], xy[1], -h_c])
    return Box(name=name, cls=cls, center=center, R=R, half=np.array(scale_ntu) / 2.0)


def _build_punched_face(face, p: BuildingParams, rng, name) -> list:
    """A solid wall with discrete punched windows (glass recessed in a reveal + a frame)."""
    boxes = []
    S2d = np.array(face.start); n2d = np.array(face.normal[:2]); t2d = np.array(face.tangent[:2])
    R = _face_R(face)
    w = face.width; total_h = p.n_floors * p.floor_height
    s0 = 0.08; w_use = max(w - 2 * s0, 0.5)
    n_cols = max(1, int(round(w_use / p.module_width)))
    col_w = w_use / n_cols
    win_w = min(p.window_w, col_w - 0.4)                  # keep a wall margin each side
    win_h = min(p.window_h, p.floor_height - 0.8)
    out_wall = p.frame_depth / 2                          # wall + window frame at the facade front
    out_glass = -p.recess - _GLASS_THICK / 2             # glass recessed in the reveal
    mf = p.mullion_face
    gpref = _GLASS_PREFIX[p.glass_class]
    for c in range(n_cols):
        s_c = s0 + (c + 0.5) * col_w
        for fl in range(p.n_floors):
            base_h = fl * p.floor_height
            sill = (p.floor_height - win_h) / 2
            wcz = base_h + sill + win_h / 2; wb = wcz - win_h / 2; wt = wcz + win_h / 2
            boxes.append(_box(name(gpref), p.glass_class, S2d, n2d, t2d, s_c, wcz, out_glass,
                              (_GLASS_THICK, win_w, win_h), R))
            # window frame: 4 bars around the opening
            boxes.append(_box(name("Frame_"), M.METAL_FRAME, S2d, n2d, t2d, s_c, wb, out_wall, (p.frame_depth, win_w + mf, mf), R))
            boxes.append(_box(name("Frame_"), M.METAL_FRAME, S2d, n2d, t2d, s_c, wt, out_wall, (p.frame_depth, win_w + mf, mf), R))
            boxes.append(_box(name("Frame_"), M.METAL_FRAME, S2d, n2d, t2d, s_c - win_w / 2, wcz, out_wall, (p.frame_depth, mf, win_h), R))
            boxes.append(_box(name("Frame_"), M.METAL_FRAME, S2d, n2d, t2d, s_c + win_w / 2, wcz, out_wall, (p.frame_depth, mf, win_h), R))
            # wall tiling the cell around the window (above / below / left / right)
            side_w = (col_w - win_w) / 2
            above_h = (base_h + p.floor_height) - wt
            below_h = wb - base_h
            if above_h > 0.05:
                boxes.append(_box(name("Wall_"), M.WALL, S2d, n2d, t2d, s_c, (wt + base_h + p.floor_height) / 2, out_wall, (p.frame_depth, col_w, above_h), R))
            if below_h > 0.05:
                boxes.append(_box(name("Wall_"), M.WALL, S2d, n2d, t2d, s_c, (base_h + wb) / 2, out_wall, (p.frame_depth, col_w, below_h), R))
            if side_w > 0.05:
                boxes.append(_box(name("Wall_"), M.WALL, S2d, n2d, t2d, s_c - (win_w + col_w) / 4, wcz, out_wall, (p.frame_depth, side_w, win_h), R))
                boxes.append(_box(name("Wall_"), M.WALL, S2d, n2d, t2d, s_c + (win_w + col_w) / 4, wcz, out_wall, (p.frame_depth, side_w, win_h), R))
    return boxes


def _build_face(face, p: BuildingParams, rng, name) -> list:
    boxes = _build_punched_face(face, p, rng, name) if p.punched else _build_curtain_face(face, p, rng, name)
    boxes += _add_horizontals(face, p, rng, name)         # bands / parapet / plinth (both face types)
    return boxes


def _build_curtain_face(face, p: BuildingParams, rng, name) -> list:
    boxes = []
    S2d = np.array(face.start); n2d = np.array(face.normal[:2]); t2d = np.array(face.tangent[:2])
    R = _face_R(face)
    w = face.width
    total_h = p.n_floors * p.floor_height
    s0 = 0.08                                             # inset content off the shared corners (covered by the corner post)
    w_use = max(w - 2 * s0, 0.5)
    n_cols = max(1, int(round(w_use / p.module_width)))
    col_w = w_use / n_cols
    full_wall = rng.random() < p.full_wall_face_prob
    col_is_wall = [full_wall or (rng.random() < p.wall_col_prob) for _ in range(n_cols)]

    # Depth model (matches a real curtain wall): the glass sits ~flush at the facade plane and
    # the mullion/transom grid protrudes `frame_depth` (1-12 cm) IN FRONT of it. So the frame
    # front stands exactly `frame_depth` proud of the glass -- realistic to the eye AND enough
    # range separation (>=2 cm) for the lidar to tell frame from glass. No deep artificial recess.
    out_frame = p.frame_depth / 2                         # frame spans [0, frame_depth] outward
    out_glass = -_GLASS_THICK / 2                         # glass front flush with the facade plane
    out_spand = -0.025                                    # spandrel front ~flush too
    for c in range(n_cols + 1):                           # mullions (vertical) at column boundaries
        boxes.append(_box(name("Frame_"), M.METAL_FRAME, S2d, n2d, t2d,
                          s0 + c * col_w, total_h / 2, out_frame,
                          (p.frame_depth, p.mullion_face, total_h), R))
    for fl in range(p.n_floors + 1):                      # transoms (horizontal) at floor lines
        boxes.append(_box(name("Frame_"), M.METAL_FRAME, S2d, n2d, t2d,
                          s0 + w_use / 2, fl * p.floor_height, out_frame,
                          (p.frame_depth, w_use, p.mullion_face), R))

    gw = col_w * p.window_width_frac                      # vision pane width (<= bay)
    side = (col_w - gw) / 2                               # spandrel infill each side when inset
    for c in range(n_cols):
        s_c = s0 + (c + 0.5) * col_w
        if col_is_wall[c]:                                # solid wall column, at the facade front
            boxes.append(_box(name("Wall_"), M.WALL, S2d, n2d, t2d, s_c, total_h / 2,
                              out_frame, (p.frame_depth, col_w, total_h), R))
            continue
        # Each floor is FULLY filled: a spandrel band (the sill) below + vision glass above,
        # both recessed behind the frame grid.
        sill_h = (1.0 - p.vision_fraction) * p.floor_height
        vision_h = p.vision_fraction * p.floor_height
        for fl in range(p.n_floors):
            base_h = fl * p.floor_height
            if sill_h > 0.05:
                boxes.append(_box(name("Spandrel_"), M.SPANDREL, S2d, n2d, t2d,
                                  s_c, base_h + sill_h / 2, out_spand, (0.05, col_w, sill_h), R))
                if p.intermediate_transom:                  # frame at the vision/spandrel boundary
                    boxes.append(_box(name("Frame_"), M.METAL_FRAME, S2d, n2d, t2d,
                                      s_c, base_h + sill_h, out_frame,
                                      (p.frame_depth, col_w, p.mullion_face), R))
            vcz = base_h + sill_h + vision_h / 2
            boxes.append(_box(name(_GLASS_PREFIX[p.glass_class]), p.glass_class, S2d, n2d, t2d,
                              s_c, vcz, out_glass, (_GLASS_THICK, gw, vision_h), R))
            if side > 0.03:                              # inset window -> spandrel infill on each side
                for sgn in (-1.0, 1.0):
                    boxes.append(_box(name("Spandrel_"), M.SPANDREL, S2d, n2d, t2d,
                                      s_c + sgn * (gw + side) / 2, vcz, out_spand,
                                      (0.05, side, vision_h), R))
    return boxes


def _add_horizontals(face, p: BuildingParams, rng, name) -> list:
    """Protruding solid (WALL) horizontals: floor-line string-course bands, a roof parapet and a
    base plinth -- the cornices/ledges real buildings carry, standing proud of the facade."""
    out = []
    S2d = np.array(face.start); n2d = np.array(face.normal[:2]); t2d = np.array(face.tangent[:2])
    R = _face_R(face)
    total_h = p.n_floors * p.floor_height
    s0 = 0.08; w_use = max(face.width - 2 * s0, 0.5); sc = s0 + w_use / 2

    def band(prefix, h_c, depth, height):
        out.append(_box(name(prefix), M.WALL, S2d, n2d, t2d, sc, h_c, depth / 2,
                        (depth, w_use, height), R))

    if p.band_prob > 0 and p.band_depth > 0:
        for fl in range(1, p.n_floors):                  # interior floor lines only
            if rng.random() < p.band_prob:
                band("Wall_Band_", fl * p.floor_height, p.band_depth, 0.30)
    if p.parapet:
        band("Wall_Parapet_", total_h + 0.45, max(p.frame_depth, 0.12), 0.90)
    if p.plinth:
        band("Wall_Plinth_", 0.45, max(2 * p.frame_depth, 0.20), 0.90)
    return out


def build_building(footprint: Footprint, p: BuildingParams, rng) -> Building:
    name = _Namer()
    boxes = []
    total_h = p.n_floors * p.floor_height
    for face in footprint.faces:
        boxes += _build_face(face, p, rng, name)
    # corner posts: a chunky square mullion at each footprint vertex. It both makes adjacent
    # faces meet cleanly AND hides the inset (s0) ends of the transoms/mullions, so even a few-cm
    # spawn-scale residual can't leave a bar poking past the corner into open air.
    for v in footprint.vertices:
        boxes.append(Box(name=name("Frame_"), cls=M.METAL_FRAME,
                         center=np.array([float(v[0]), float(v[1]), -total_h / 2.0]),
                         R=np.eye(3), half=np.array([0.08, 0.08, total_h / 2.0])))
    xs = [v[0] for v in footprint.vertices]; ys = [v[1] for v in footprint.vertices]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    # ground slab around the base (z = 0), axis-aligned, top face up
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    boxes.append(Box(name="Ground_Plane", cls=M.GROUND,
                     center=np.array([cx, cy, 0.0]), R=np.eye(3),
                     half=np.array([span, span, 0.02])))
    return Building(boxes=boxes, params=p, footprint=footprint)


def sample_building(rng, region: str = "gcc", square_prob: float = 0.6) -> Building:
    fp = sample_footprint(rng, square_prob=square_prob)
    return build_building(fp, sample_building_params(rng, region), rng)
