"""Tests for building footprints (closed polygons) and their per-face outward normals.

World frame is NED: X=North, Y=East, Z=Down (up = -Z). Footprints live in the X-Y
plane; faces are vertical walls, so their outward normals are horizontal (nz = 0).
"""
import numpy as np

from buildings.footprint import sample_footprint, footprint_from_vertices


def test_square_has_four_faces():
    fp = footprint_from_vertices([(-5, -4), (5, -4), (5, 4), (-5, 4)])
    assert len(fp.faces) == 4


def test_every_face_normal_points_away_from_centroid():
    fp = footprint_from_vertices([(-5, -4), (5, -4), (5, 4), (-5, 4)])
    centroid = np.mean(fp.vertices, axis=0)
    for f in fp.faces:
        mid = (np.array(f.start) + np.array(f.end)) / 2
        outward = mid - centroid
        n2 = np.array(f.normal[:2])
        assert n2 @ outward > 0, f"normal {f.normal} not outward at {mid}"


def test_face_normals_are_horizontal_unit_vectors():
    fp = footprint_from_vertices([(-5, -4), (5, -4), (5, 4), (-5, 4)])
    for f in fp.faces:
        assert abs(f.normal[2]) < 1e-12
        assert abs(np.linalg.norm(f.normal) - 1.0) < 1e-9


def test_face_width_matches_edge_length():
    fp = footprint_from_vertices([(-5, -4), (5, -4), (5, 4), (-5, 4)])
    widths = sorted(f.width for f in fp.faces)
    assert np.allclose(widths, [8, 8, 10, 10])      # edges of a 10x8 rectangle


def test_square_normals_are_axis_aligned():
    # A rectangle's faces must face exactly +X/-X/+Y/-Y (zero rotation-convention risk).
    fp = footprint_from_vertices([(-5, -4), (5, -4), (5, 4), (-5, 4)])
    for f in fp.faces:
        n = np.array(f.normal)
        assert np.isclose(np.abs(n).max(), 1.0)     # one axis is ±1
        assert np.isclose(np.abs(n).sum(), 1.0)     # the others are 0


def test_sample_square_dimensions_in_range_and_kind():
    fp = sample_footprint(np.random.default_rng(0), square_prob=1.0)
    assert fp.kind == "square"
    xs = [v[0] for v in fp.vertices]; ys = [v[1] for v in fp.vertices]
    assert 8.0 <= (max(xs) - min(xs)) <= 30.0
    assert 8.0 <= (max(ys) - min(ys)) <= 30.0


def test_square_probability_is_respected():
    rng = np.random.default_rng(1)
    kinds = [sample_footprint(rng, square_prob=0.6).kind for _ in range(400)]
    frac_square = sum(k == "square" for k in kinds) / len(kinds)
    assert 0.5 < frac_square < 0.7


def test_polygon_faces_still_outward_and_unit():
    # Non-square footprints must keep outward, unit normals -- checked robustly (point-in-polygon,
    # valid for the concave L/U/T shapes) across many seeds, not by a convex-only centroid test.
    from matplotlib.path import Path
    for seed in range(20):
        fp = sample_footprint(np.random.default_rng(seed), square_prob=0.0)
        assert fp.kind != "square"
        assert len(fp.faces) >= 5
        poly = Path(np.asarray(fp.vertices, float))
        for f in fp.faces:
            mid = (np.array(f.start) + np.array(f.end)) / 2
            n = np.array(f.normal[:2])
            assert abs(np.linalg.norm(f.normal) - 1.0) < 1e-9
            assert not poly.contains_point(mid + 0.05 * n)
            assert poly.contains_point(mid - 0.05 * n)


def test_nonsquare_footprint_is_axis_aligned_rectilinear():
    # Default non-square footprints are axis-aligned rectilinear shapes (L/U/T): every face still
    # faces +/-X or +/-Y, so they spawn correctly with NO rotated-box work. (Angled is opt-in.)
    rng = np.random.default_rng(0)
    for _ in range(50):
        fp = sample_footprint(rng, square_prob=0.0)
        assert fp.kind == "polygon"
        assert len(fp.faces) >= 6                       # L/U/T have >=6 edges
        for f in fp.faces:
            n = np.array(f.normal)
            assert np.isclose(np.abs(n).max(), 1.0) and np.isclose(np.abs(n).sum(), 1.0), \
                f"face normal {f.normal} is not axis-aligned"


def _u_shape():
    # A U opening toward +Y (concave). Vertices wound CCW.
    return [(0, 0), (9, 0), (9, 9), (6, 9), (6, 3), (3, 3), (3, 9), (0, 9)]


def test_concave_footprint_normals_point_outward():
    # Robust invariant for ANY simple polygon (incl. concave L/U/T): a hair along a face's
    # outward normal must LEAVE the polygon, and the other way must stay INSIDE. The old
    # centroid heuristic flips the normal on the U's inner notch floor; winding order fixes it.
    from matplotlib.path import Path
    fp = footprint_from_vertices(_u_shape(), kind="polygon")
    poly = Path(np.asarray(fp.vertices, float))
    for f in fp.faces:
        mid = (np.array(f.start) + np.array(f.end)) / 2
        n = np.array(f.normal[:2])
        assert not poly.contains_point(mid + 0.05 * n), f"normal {f.normal} points inward at {mid}"
        assert poly.contains_point(mid - 0.05 * n), f"inward step left the polygon at {mid}"
