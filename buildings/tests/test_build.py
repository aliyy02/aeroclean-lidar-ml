"""Tests for building geometry invariants (oriented boxes from a footprint)."""
from dataclasses import replace

import numpy as np

from forward_model import materials as M
from buildings.footprint import footprint_from_vertices
from buildings.build import sample_building_params, build_building, sample_building

_SQUARE = [(-8, -7), (8, -7), (8, 7), (-8, 7)]


def _curtain_params(seed=0, **over):
    p = sample_building_params(np.random.default_rng(seed), "gcc")
    base = dict(system="S1", punched=False, vision_fraction=0.7, wall_col_prob=0.0,
                full_wall_face_prob=0.0, intermediate_transom=False, band_prob=0.0,
                parapet=False, plinth=False)
    base.update(over)
    return replace(p, **base)

GLASS = {M.GLASS_CLEAR, M.GLASS_COATED, M.GLASS_LOWE}


def square_building(seed=0, region="gcc"):
    rng = np.random.default_rng(seed)
    fp = footprint_from_vertices([(-8, -7), (8, -7), (8, 7), (-8, 7)], kind="square")
    return build_building(fp, sample_building_params(rng, region), rng)


def test_building_contains_all_element_classes():
    classes = {b.cls for b in square_building().boxes}
    assert M.METAL_FRAME in classes
    assert M.SPANDREL in classes
    assert M.GROUND in classes
    assert classes & GLASS


def test_all_boxes_have_orthonormal_rotation_and_positive_extents():
    for b in square_building().boxes:
        assert np.allclose(b.R.T @ b.R, np.eye(3), atol=1e-9)
        assert np.all(b.half > 0)


def test_square_building_boxes_are_axis_aligned():
    # Zero rotation-convention risk: every R entry is 0 or +/-1.
    for b in square_building().boxes:
        assert np.allclose(np.abs(b.R), np.round(np.abs(b.R)), atol=1e-9)
        assert set(np.round(b.R.flatten(), 6)).issubset({-1.0, 0.0, 1.0})


def test_names_roundtrip_through_classifier():
    for b in square_building().boxes:
        assert M.material_for(b.name) == b.cls, b.name


def test_one_glass_type_per_building():
    gclasses = {b.cls for b in square_building().boxes if b.cls in GLASS}
    assert len(gclasses) == 1


def test_box_front_faces_point_away_from_footprint_centroid():
    # Facade panels (glass/spandrel) must face outward. Corner posts, the roof and ground
    # are special (no single outward face), so they're excluded.
    bld = square_building()
    centroid = np.mean(bld.footprint.vertices, axis=0)
    for b in bld.boxes:
        if b.cls not in (M.SPANDREL, M.GLASS_CLEAR, M.GLASS_COATED, M.GLASS_LOWE):
            continue
        front_normal = b.R[:, 0]                          # +n axis = outward face
        outward = b.center[:2] - centroid
        assert front_normal[:2] @ outward > 0


def test_sampled_building_floor_count_in_range():
    bld = sample_building(np.random.default_rng(2), square_prob=1.0)
    assert 5 <= bld.params.n_floors <= 12


def test_window_width_frac_narrows_the_glass():
    # window_width_frac < 1 must make the vision glass narrower than a full-bay pane.
    fp = footprint_from_vertices(_SQUARE, kind="square")

    def widest_glass(frac):
        p = _curtain_params(window_width_frac=frac)
        bld = build_building(fp, p, np.random.default_rng(0))
        return max(b.half[1] * 2 for b in bld.boxes if b.cls in GLASS)

    assert widest_glass(0.6) < 0.7 * widest_glass(1.0)


def test_horizontal_bands_protrude_and_span_the_face():
    # A floor-line band is a solid ledge standing well proud of the frame and running across the face.
    fp = footprint_from_vertices(_SQUARE, kind="square")
    p = _curtain_params(band_prob=1.0, band_depth=0.3, frame_depth=0.05)
    bld = build_building(fp, p, np.random.default_rng(2))
    bands = [b for b in bld.boxes if "Band" in b.name]
    assert bands, "no bands generated"
    for b in bands:
        assert b.cls == M.WALL and M.material_for(b.name) == M.WALL
        assert b.half[0] * 2 > p.frame_depth + 0.05      # protrudes beyond the frame
        assert b.half[1] * 2 > 6.0                        # spans most of a ~15.8 m face


def test_parapet_and_plinth_extend_above_and_below():
    fp = footprint_from_vertices(_SQUARE, kind="square")
    p = _curtain_params(parapet=True, plinth=True, n_floors=6, floor_height=4.0)
    bld = build_building(fp, p, np.random.default_rng(0))
    total_h = 24.0
    top = max(-b.center[2] + b.half[2] for b in bld.boxes)        # highest world point
    assert top > total_h + 0.3, "parapet not above the top floor"
    assert any("Plinth" in b.name for b in bld.boxes)


def test_rectilinear_building_spawns_without_rotated_boxes():
    # L/U/T footprints are axis-aligned, so EVERY box must still be axis-aligned (R entries in
    # {0,+-1}) -> it spawns with the identity-quaternion spawner, no rotated-box convention needed.
    from buildings.footprint import sample_footprint
    for seed in range(10):
        rng = np.random.default_rng(seed)
        fp = sample_footprint(rng, square_prob=0.0)               # an L/U/T
        assert fp.kind == "polygon"
        bld = build_building(fp, sample_building_params(rng, "lebanon"), rng)
        classes = {b.cls for b in bld.boxes}
        assert M.METAL_FRAME in classes and (classes & GLASS)     # real facade, not empty
        for b in bld.boxes:
            assert set(np.round(b.R.flatten(), 6)).issubset({-1.0, 0.0, 1.0}), (seed, b.name)
