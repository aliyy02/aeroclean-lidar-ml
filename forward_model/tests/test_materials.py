"""Tests for actor-name -> material/class id mapping."""
from forward_model.materials import (
    WALL, METAL_FRAME, GLASS_CLEAR, GLASS_COATED, GLASS_LOWE, SPANDREL, GROUND, OTHER,
    material_for,
)


def test_each_prefix_maps_to_its_class():
    assert material_for("Wall_North") == WALL
    assert material_for("Frame_Top") == METAL_FRAME
    assert material_for("GlassClear_3") == GLASS_CLEAR
    assert material_for("GlassCoated_12") == GLASS_COATED
    assert material_for("GlassLowE_0") == GLASS_LOWE
    assert material_for("Spandrel_Floor2") == SPANDREL
    assert material_for("Ground_Plane") == GROUND


def test_unknown_name_is_other():
    assert material_for("StreetLamp_7") == OTHER
    assert material_for("") == OTHER


def test_glass_subclasses_do_not_collide():
    # The longer-specific prefixes must win; none should fall back to a generic match.
    assert material_for("GlassLowE_1") != material_for("GlassClear_1")
    assert material_for("GlassCoated_1") != material_for("GlassClear_1")


def test_legacy_phase0_names_still_classify():
    # Continuity with phase0/test_stdlidar_cv.py scene actor names.
    assert material_for("TestFrame_Top") == METAL_FRAME
    assert material_for("TestGlass_Pane") == GLASS_CLEAR
