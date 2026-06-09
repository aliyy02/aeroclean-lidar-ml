"""Per-type material archetypes + diversification sampler.

Each archetype carries RANGES for the reflectance/detect params; the sampler draws a
concrete material per window/surface so sim training spans the real material spectrum.
"""
import numpy as np

from forward_model import materials as M
from forward_model.material_library import ARCHETYPES, sample_material, sample_glass


def test_all_params_sampled_in_range():
    rng = np.random.default_rng(0)
    for name, arc in ARCHETYPES.items():
        mat = sample_material(rng, name)
        for fld in ("a", "g", "s", "b", "m", "p_floor", "cone", "tau"):
            lo, hi = getattr(arc, fld)
            assert lo - 1e-9 <= getattr(mat, fld) <= hi + 1e-9, f"{name}.{fld} out of range"


def test_clear_glass_is_specular_and_see_through():
    rng = np.random.default_rng(1)
    mat = sample_material(rng, "glass_clear")
    assert mat.is_glass and mat.p_floor < 0.2 and mat.tau > 0.5   # mirror-ish, transmits


def test_coated_lowe_glass_returns_everywhere():
    rng = np.random.default_rng(2)
    mat = sample_material(rng, "glass_lowE")          # our data-anchored type
    assert mat.is_glass and mat.p_floor > 0.7         # returns at all angles


def test_opaque_has_no_transmission_and_returns_always():
    rng = np.random.default_rng(3)
    for name in ("wall", "metal_frame", "ground"):
        mat = sample_material(rng, name)
        assert not mat.is_glass and mat.tau == 0.0 and mat.p_floor == 1.0


def test_sampler_reproducible_with_seed():
    a = sample_material(np.random.default_rng(7), "glass_lowE")
    b = sample_material(np.random.default_rng(7), "glass_lowE")
    assert a.a == b.a and a.b == b.b and a.m == b.m


def test_sample_glass_picks_a_glass_type():
    rng = np.random.default_rng(4)
    name, mat = sample_glass(rng)
    assert name.startswith("glass_") and mat.is_glass


def test_train_labels_map_to_three_classes():
    assert ARCHETYPES["glass_lowE"].train_label == M.GLASS
    assert ARCHETYPES["metal_frame"].train_label == M.NOT_GLASS
    assert ARCHETYPES["ground"].train_label == M.GROUND_3
