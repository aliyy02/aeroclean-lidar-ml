"""Map a per-point UE5 actor-name string to a material / class id.

Standard Lidar (SensorType 6) returns a `groundtruth` list of actor-name strings,
one per point. The scene generator names actors with class-prefixes (see the table
below), so classification is a trivial longest-prefix match -- no RGB decoding.

Material ids double as the synthesis material (each gets its own a,b,m in
constants.py) AND the per-point class label written into the .npz training data.
A coarser training-label grouping can be applied downstream via TRAIN_LABEL_OF.
"""
from __future__ import annotations

# Canonical class / synthesis-material ids.
WALL = 0
METAL_FRAME = 1
GLASS_CLEAR = 2
GLASS_COATED = 3
GLASS_LOWE = 4
SPANDREL = 5
GROUND = 6
OTHER = 7

CLASS_NAMES = {
    WALL: "wall",
    METAL_FRAME: "metal_frame",
    GLASS_CLEAR: "glass_clear",
    GLASS_COATED: "glass_coated",
    GLASS_LOWE: "glass_lowE",
    SPANDREL: "spandrel",
    GROUND: "ground",
    OTHER: "other",
}

# Actor-name prefix -> class id. Includes the legacy phase0 Test* names for
# continuity with phase0/test_stdlidar_cv.py scenes. Matched longest-prefix-first
# so that e.g. "GlassClear_" wins and never falls through to a shorter prefix.
_PREFIX_TO_CLASS = {
    "Wall_": WALL,
    "Frame_": METAL_FRAME,
    "GlassClear_": GLASS_CLEAR,
    "GlassCoated_": GLASS_COATED,
    "GlassLowE_": GLASS_LOWE,
    "Spandrel_": SPANDREL,
    "Ground_": GROUND,
    "Ground": GROUND,
    # legacy phase0 names
    "TestFrame_": METAL_FRAME,
    "TestGlass_": GLASS_CLEAR,
}

# Prefixes sorted longest-first so specific names beat generic ones.
_PREFIXES_SORTED = sorted(_PREFIX_TO_CLASS, key=len, reverse=True)


def material_for(name: str) -> int:
    """Return the class id for a single actor-name string (OTHER if unrecognized)."""
    for prefix in _PREFIXES_SORTED:
        if name.startswith(prefix):
            return _PREFIX_TO_CLASS[prefix]
    return OTHER


# ---------------------------------------------------------------------------
# Coarse 3-class TRAINING labels (what the cleaning drone actually needs):
#   GLASS     -> a cleanable surface: vision glass (any type) AND spandrel
#                (spandrel panels are glass/opacified and get washed too)
#   NOT_GLASS -> everything to avoid touching: frames, walls, poles, etc.
#   GROUND    -> the floor
# The fine 8-class id stays in the raw .npz (the forward model needs glass
# sub-type for intensity); collapse to these only at training time.
NOT_GLASS = 0
GLASS = 1
GROUND_3 = 2

TRAIN3_NAMES = {NOT_GLASS: "not_glass", GLASS: "glass", GROUND_3: "ground"}

_FINE_TO_TRAIN3 = {
    WALL: NOT_GLASS, METAL_FRAME: NOT_GLASS, OTHER: NOT_GLASS,
    GLASS_CLEAR: GLASS, GLASS_COATED: GLASS, GLASS_LOWE: GLASS, SPANDREL: GLASS,
    GROUND: GROUND_3,
}


def train3_label(fine_id: int) -> int:
    """Collapse a fine 8-class id to the coarse 3-class training label."""
    return _FINE_TO_TRAIN3[int(fine_id)]
