"""Per-type material archetypes + a diversification sampler.

Each archetype gives RANGES for the revised model's params; `sample_material` draws a
concrete material per surface so synthetic training spans the real spectrum (specular
<-> glossy <-> diffuse, see-through <-> opaque). Only `glass_lowE` is data-anchored
(the L6 coated glass); the others are physics+literature predictions. See
`calibration/GLASS_MODEL_DIVERSIFICATION.md` and `GLASS_BEHAVIOR_L6.md`.

params: a diffuse, g flatness, s grazing-lift, b/m specular burst (reflectance.py);
        p_floor/cone return-chance, tau see-through (detect.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from . import materials as M

R = Tuple[float, float]


@dataclass
class Archetype:
    a: R; g: R; s: R; b: R; m: R; p_floor: R; cone: R; tau: R
    is_glass: bool
    train_label: int
    n_index: float = 1.5


@dataclass
class Material:
    name: str
    a: float; g: float; s: float; b: float; m: float
    p_floor: float; cone: float; tau: float
    n_index: float
    is_glass: bool
    train_label: int


# --- opaque (data-anchored: all flatter than cosine; metal lifts at grazing) ---
ARCHETYPES: Dict[str, Archetype] = {
    "wall":        Archetype(a=(0.6, 1.0), g=(0.30, 0.60), s=(0.0, 0.05), b=(0.0, 0.05),
                             m=(0.20, 0.40), p_floor=(1.0, 1.0), cone=(0.3, 0.3),
                             tau=(0.0, 0.0), is_glass=False, train_label=M.NOT_GLASS),
    "metal_frame": Archetype(a=(0.5, 0.9), g=(0.35, 0.55), s=(0.10, 0.30), b=(0.10, 0.30),
                             m=(0.05, 0.20), p_floor=(1.0, 1.0), cone=(0.3, 0.3),
                             tau=(0.0, 0.0), is_glass=False, train_label=M.NOT_GLASS),
    "ground":      Archetype(a=(0.7, 1.1), g=(0.35, 0.55), s=(0.0, 0.05), b=(0.0, 0.03),
                             m=(0.20, 0.40), p_floor=(1.0, 1.0), cone=(0.3, 0.3),
                             tau=(0.0, 0.0), is_glass=False, train_label=M.GROUND_3),
    "spandrel":    Archetype(a=(0.4, 0.7), g=(0.30, 0.60), s=(0.0, 0.10), b=(0.0, 0.10),
                             m=(0.20, 0.40), p_floor=(1.0, 1.0), cone=(0.3, 0.3),
                             tau=(0.0, 0.0), is_glass=False, train_label=M.GLASS),
    # --- glass spectrum (glass_lowE anchored to L6 data; others predicted) ---
    "glass_clear":      Archetype(a=(0.02, 0.08), g=(0.20, 0.50), s=(0.30, 0.60),
                                  b=(0.10, 0.30), m=(0.02, 0.05), p_floor=(0.02, 0.15),
                                  cone=(0.03, 0.09), tau=(0.70, 0.90), is_glass=True,
                                  train_label=M.GLASS),
    "glass_tinted":     Archetype(a=(0.02, 0.06), g=(0.20, 0.50), s=(0.20, 0.50),
                                  b=(0.10, 0.30), m=(0.02, 0.06), p_floor=(0.05, 0.20),
                                  cone=(0.05, 0.14), tau=(0.20, 0.50), is_glass=True,
                                  train_label=M.GLASS),
    "glass_lowE":       Archetype(a=(0.35, 0.55), g=(0.50, 0.80), s=(0.15, 0.40),
                                  b=(0.40, 0.60), m=(0.07, 0.15), p_floor=(0.80, 1.00),
                                  cone=(0.20, 0.40), tau=(0.10, 0.40), is_glass=True,
                                  train_label=M.GLASS),
    "glass_reflective": Archetype(a=(0.10, 0.30), g=(0.30, 0.60), s=(0.40, 0.80),
                                  b=(0.55, 0.85), m=(0.04, 0.10), p_floor=(0.40, 0.80),
                                  cone=(0.08, 0.25), tau=(0.00, 0.20), is_glass=True,
                                  train_label=M.GLASS),
    "glass_frosted":    Archetype(a=(0.40, 0.70), g=(0.40, 0.70), s=(0.00, 0.10),
                                  b=(0.00, 0.15), m=(0.30, 0.70), p_floor=(0.90, 1.00),
                                  cone=(0.30, 0.50), tau=(0.10, 0.40), is_glass=True,
                                  train_label=M.GLASS),
}

# how often each glass type appears (tune per region); used by sample_glass
GLASS_PREVALENCE = {"glass_clear": 0.30, "glass_lowE": 0.30, "glass_tinted": 0.15,
                    "glass_reflective": 0.10, "glass_frosted": 0.15}

_FIELDS = ("a", "g", "s", "b", "m", "p_floor", "cone", "tau")


def sample_material(rng: np.random.Generator, name: str) -> Material:
    """Draw a concrete Material from an archetype's ranges (uniform per field)."""
    arc = ARCHETYPES[name]
    vals = {f: float(rng.uniform(*getattr(arc, f))) for f in _FIELDS}
    return Material(name=name, n_index=arc.n_index, is_glass=arc.is_glass,
                    train_label=arc.train_label, **vals)


def sample_glass(rng: np.random.Generator):
    """Pick a glass type by prevalence, then sample its params. Returns (name, Material)."""
    names = list(GLASS_PREVALENCE)
    w = np.array([GLASS_PREVALENCE[n] for n in names], dtype=float)
    name = names[int(rng.choice(len(names), p=w / w.sum()))]
    return name, sample_material(rng, name)
