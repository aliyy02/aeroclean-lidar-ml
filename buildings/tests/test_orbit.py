"""Tests for the orbit capture-pose grid + biased standoff/orientation sampling."""
import numpy as np

from buildings.footprint import footprint_from_vertices
from buildings.build import build_building, sample_building_params
from buildings.orbit import face_grid, sample_pose, capture_poses, grid_spacing


def a_face():
    fp = footprint_from_vertices([(-8, -7), (8, -7), (8, 7), (-8, 7)])
    return fp.faces[0]


def test_grid_spacing_coarsens_for_large_faces():
    assert grid_spacing(area=100.0) == 1.0
    assert grid_spacing(area=800.0) == 1.5


def test_face_grid_points_lie_on_the_face_within_bounds():
    face = a_face()
    pts = face_grid(face, total_h=20.0, spacing=1.0)
    assert len(pts) > 0
    for fp, n, up in pts:
        # on the face plane: displacement from start has no outward-normal component
        d = fp - np.array([face.start[0], face.start[1], 0.0])
        assert abs(d @ np.array(face.normal)) < 1e-6
        assert -20.0 <= fp[2] <= 0.0                       # within building height (z up = negative)


def test_standoff_is_biased_to_one_to_two_metres():
    rng = np.random.default_rng(0)
    ds = [sample_pose(rng, np.zeros(3), np.array([1.0, 0, 0])).standoff for _ in range(2000)]
    assert np.all((np.array(ds) >= 1.0) & (np.array(ds) <= 4.0))
    assert np.mean([1.0 <= d <= 2.0 for d in ds]) > 0.6     # majority in 1-2 m


def test_pitch_is_biased_down_zero_to_minus_twenty():
    rng = np.random.default_rng(0)
    ps = [sample_pose(rng, np.zeros(3), np.array([1.0, 0, 0])).pitch for _ in range(2000)]
    assert np.all((np.array(ps) >= -40) & (np.array(ps) <= 40))
    assert np.mean([-20 <= p <= 0 for p in ps]) > 0.55
    assert np.mean(ps) < -3                                 # net downward


def test_roll_and_yaw_centre_near_zero():
    rng = np.random.default_rng(0)
    poses = [sample_pose(rng, np.zeros(3), np.array([1.0, 0, 0])) for _ in range(2000)]
    assert abs(np.mean([p.roll for p in poses])) < 3
    assert abs(np.mean([p.yaw_dev for p in poses])) < 3


def test_sensor_sits_standoff_out_along_the_outward_normal():
    rng = np.random.default_rng(0)
    face_pt = np.array([3.0, 1.0, -5.0]); n = np.array([1.0, 0, 0])
    p = sample_pose(rng, face_pt, n)
    np.testing.assert_allclose(p.sensor_pos, face_pt + p.standoff * n, atol=1e-9)


def test_capture_poses_cover_all_faces_with_k_per_cell():
    rng = np.random.default_rng(0)
    fp = footprint_from_vertices([(-8, -7), (8, -7), (8, 7), (-8, 7)])
    bld = build_building(fp, sample_building_params(rng, "gcc"), rng)
    poses = list(capture_poses(bld, rng, k=4))
    assert len(poses) > 0
    assert set(p.face_idx for p in poses) == set(range(len(fp.faces)))


def test_capture_poses_budget_caps_count_and_spreads_across_faces():
    # A per-building budget must cap the number of shots AND draw them from across the faces
    # (not front-load face 0), so a small dataset still sees the whole building.
    rng = np.random.default_rng(0)
    fp = footprint_from_vertices([(-8, -7), (8, -7), (8, 7), (-8, 7)])
    bld = build_building(fp, sample_building_params(rng, "gcc"), rng)
    poses = list(capture_poses(bld, np.random.default_rng(1), k=4, budget=40))
    assert len(poses) == 40
    assert len(set(p.face_idx for p in poses)) >= 3      # spread, not all on one face
