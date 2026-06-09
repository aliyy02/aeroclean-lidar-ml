"""Tests for the named placeholder constants + calibrated-YAML override loader."""
import textwrap

from forward_model import materials as M
from forward_model.constants import default_constants, load_constants


def test_default_has_all_eight_materials():
    c = default_constants()
    for mid in (M.WALL, M.METAL_FRAME, M.GLASS_CLEAR, M.GLASS_COATED,
                M.GLASS_LOWE, M.SPANDREL, M.GROUND, M.OTHER):
        assert mid in c.materials


def test_only_glass_is_cone_gated():
    c = default_constants()
    assert c.materials[M.GLASS_CLEAR].is_glass
    assert c.materials[M.GLASS_COATED].is_glass
    assert c.materials[M.GLASS_LOWE].is_glass
    assert not c.materials[M.WALL].is_glass
    assert not c.materials[M.METAL_FRAME].is_glass
    assert not c.materials[M.GROUND].is_glass


def test_prior_specular_increases_with_coating():
    # Physical sanity of the priors: b ~ Fresnel, so low-E > coated > clear.
    c = default_constants()
    assert (c.materials[M.GLASS_LOWE].b
            > c.materials[M.GLASS_COATED].b
            > c.materials[M.GLASS_CLEAR].b)


def test_clear_glass_cone_is_narrow():
    c = default_constants()
    assert 0.0 < c.materials[M.GLASS_CLEAR].cone_half_width < 0.2   # < ~11 deg


def test_yaml_overrides_globals_and_materials(tmp_path):
    yaml_text = textwrap.dedent("""
        C: 999.0
        T: 0.5
        materials:
          glass_clear:
            b: 0.123
    """)
    p = tmp_path / "constants_calibrated.yaml"
    p.write_text(yaml_text)

    c = load_constants(p)
    assert c.C == 999.0
    assert c.T == 0.5
    assert c.materials[M.GLASS_CLEAR].b == 0.123
    # Untouched fields keep their default values.
    assert c.materials[M.GLASS_CLEAR].is_glass is True


def test_load_without_path_returns_defaults():
    assert load_constants(None).C == default_constants().C
