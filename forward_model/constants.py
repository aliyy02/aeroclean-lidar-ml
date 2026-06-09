"""Named forward-model constants.

PRINCIPLE (lidar_post_processing.pdf): the post-processing CODE is fixed; calibration
only swaps these NUMBERS. So every tunable lives here as a placeholder prior with a
provenance comment, and `load_constants()` lets a fitted `constants_calibrated.yaml`
override any subset. Build/generate synthetic data now; drop in the calibrated file
once the test-bed fit is done -- no code change.

Priors are seeded from legacy/materials.csv (905 nm-ish reflectances) and the
optical classes in Window Types.pdf. They are physically ORDERED (specular weight b
tracks Fresnel: low-E > coated > clear) so pre-calibration synthesis is already sane,
just not yet numerically matched to the real Unitree L2.

What calibration will pin (and what it likely cannot -- left as prior):
  FIT from the 6-DOF test bed:  a, b, m per scanned material; glass cone half-width;
       range exponent n; threshold T (via return-rate); jitter sigma.
  PRIOR (not fittable from the bed): absolute C (folded; only products matter),
       full clear/coated/low-E split if those panels aren't scanned, ghost params.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from . import materials as M


@dataclass
class MaterialParams:
    a: float                       # diffuse weight (~ rho/pi)
    b: float                       # specular weight (~ Fresnel*G); INDEPENDENT of a
    m: float                       # Beckmann roughness (specular lobe width)
    n: float = 2.0                 # range falloff exponent (2 beam-filling -> 4 thin edges)
    is_glass: bool = False         # cone-gated (specular-only return cone) if True
    cone_half_width: float = 0.0   # radians; glass direct-return cone (~ m). Used iff is_glass


@dataclass
class ForwardModelConstants:
    materials: dict
    C: float = 1.0                 # lumped system constant (folded; calibration sets scale)
    T: float = 1e-4                # detector noise floor / threshold (fit via return-rate).
                                   # Placeholder lets diffuse rho>=0.1 return to ~30 m at C=1.
    sigma: float = 0.005           # Gaussian position jitter, metres (fit on flat surfaces)
    # Optional ghost stage gating (Fong & Yan measured conditions).
    ghost_enabled: bool = False
    ghost_max_range: float = 7.5
    ghost_min_sensor_height: float = 0.7


def default_constants() -> ForwardModelConstants:
    """Placeholder priors. Calibration replaces the numbers via constants_calibrated.yaml."""
    mats = {
        # id              a      b      m     n   is_glass cone(rad)   provenance
        M.WALL:        MaterialParams(0.20, 0.00, 0.30),            # concrete/stucco rho~0.65, matte
        M.METAL_FRAME: MaterialParams(0.30, 0.15, 0.30, n=2.0),    # aluminium rho~0.90, broad NIR lobe
        M.GLASS_CLEAR: MaterialParams(0.01, 0.05, 0.03, is_glass=True,  cone_half_width=0.05),  # ~4-8%, ~3deg cone
        M.GLASS_COATED:MaterialParams(0.02, 0.30, 0.08, is_glass=True,  cone_half_width=0.10),  # solar ~20-50%
        M.GLASS_LOWE:  MaterialParams(0.02, 0.60, 0.12, is_glass=True,  cone_half_width=0.15),  # low-E ~50-80%
        M.SPANDREL:    MaterialParams(0.18, 0.05, 0.30),           # fritted/opacified, opaque
        M.GROUND:      MaterialParams(0.12, 0.00, 0.30),           # asphalt/concrete rho~0.35-0.55
        M.OTHER:       MaterialParams(0.10, 0.00, 0.30),           # generic diffuse fallback
    }
    return ForwardModelConstants(materials=mats)


_NAME_TO_ID = {name: mid for mid, name in M.CLASS_NAMES.items()}
_GLOBAL_FIELDS = {f.name for f in fields(ForwardModelConstants)} - {"materials"}
_MAT_FIELDS = {f.name for f in fields(MaterialParams)}


def load_constants(path: str | Path | None = None) -> ForwardModelConstants:
    """Return default priors, optionally overridden (partially) by a calibrated YAML file.

    YAML schema (every key optional):
        C: <float>
        T: <float>
        sigma: <float>
        materials:
          glass_clear: { a: .., b: .., m: .., n: .., cone_half_width: .. }
          metal_frame: { ... }
    """
    c = default_constants()
    if path is None:
        return c

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    for key, val in data.items():
        if key in _GLOBAL_FIELDS:
            setattr(c, key, val)
        elif key == "materials":
            for mat_name, overrides in (val or {}).items():
                mid = _NAME_TO_ID.get(mat_name)
                if mid is None:
                    raise KeyError(f"Unknown material name in calibrated YAML: {mat_name!r}")
                mp = c.materials[mid]
                for fld, fv in overrides.items():
                    if fld not in _MAT_FIELDS:
                        raise KeyError(f"Unknown material field {fld!r} for {mat_name!r}")
                    setattr(mp, fld, fv)
        else:
            raise KeyError(f"Unknown top-level key in calibrated YAML: {key!r}")
    return c
