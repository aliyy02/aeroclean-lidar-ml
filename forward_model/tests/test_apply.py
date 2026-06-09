"""Tests for the per-point forward-model orchestrator.

Ties Layers 1-4 together: geometry -> intensity (rho_eff) -> range -> cone gate (glass)
-> threshold -> jitter. Surviving points carry received power as 'intensity' and the
material id as 'label'.
"""
import numpy as np

from forward_model import materials as M
from forward_model.constants import default_constants
from forward_model.apply import apply_forward_model


def _facade_normal(n):
    """Normal facing back toward the sensor (sensor at origin, facade ahead at +x)."""
    return np.tile([-1.0, 0.0, 0.0], (n, 1))


def test_wall_directly_ahead_all_return():
    pts = np.array([[3.0, 0.0, 0.0], [3.0, 0.2, 0.1], [4.0, -0.1, 0.2]])
    names = ["Wall_N", "Wall_N", "Wall_N"]
    res = apply_forward_model(pts, names, _facade_normal(3), default_constants(),
                              rng=np.random.default_rng(0))
    assert res.xyz.shape[0] == 3
    assert np.all(res.labels == M.WALL)
    assert np.all(res.intensity > 0)


def test_glass_off_normal_is_culled_but_wall_is_not():
    # Same oblique geometry; glass outside its cone must drop, a wall must survive.
    pt = np.array([[3.0, 3.0, 0.0]])          # ~45 deg incidence to an x-facing facade
    normal = np.array([[-1.0, 0.0, 0.0]])
    c = default_constants()
    glass = apply_forward_model(pt, ["GlassClear_1"], normal, c)
    wall = apply_forward_model(pt, ["Wall_1"], normal, c)
    assert glass.xyz.shape[0] == 0            # off-cone glass: no direct return
    assert wall.xyz.shape[0] == 1


def test_glass_near_normal_returns():
    pt = np.array([[3.0, 0.0, 0.0]])          # head-on -> inside the cone
    res = apply_forward_model(pt, ["GlassCoated_1"], np.array([[-1.0, 0.0, 0.0]]),
                              default_constants())
    assert res.xyz.shape[0] == 1
    assert res.labels[0] == M.GLASS_COATED


def test_intensity_equals_received_power():
    pt = np.array([[2.0, 0.0, 0.0]])
    c = default_constants()
    res = apply_forward_model(pt, ["Wall_1"], np.array([[-1.0, 0.0, 0.0]]), c,
                              rng=np.random.default_rng(0))
    # Wall at theta=0: rho_eff = a (b=0). P_r = C * a / R^n.
    wp = c.materials[M.WALL]
    expected = c.C * wp.a / 2.0 ** wp.n
    assert np.isclose(res.intensity[0], expected)


def test_distant_dark_point_is_thresholded_out():
    c = default_constants()
    pt = np.array([[200.0, 0.0, 0.0]])        # far beyond useful range
    res = apply_forward_model(pt, ["Wall_1"], np.array([[-1.0, 0.0, 0.0]]), c)
    assert res.xyz.shape[0] == 0


def test_jitter_perturbs_surviving_points():
    pts = np.tile([3.0, 0.0, 0.0], (50, 1))
    names = ["Wall_N"] * 50
    res = apply_forward_model(pts, names, _facade_normal(50), default_constants(),
                              rng=np.random.default_rng(7))
    assert res.xyz.shape[0] == 50
    assert not np.allclose(res.xyz, pts)      # positions were jittered


def test_empty_input_returns_empty_result():
    res = apply_forward_model(np.zeros((0, 3)), [], np.zeros((0, 3)), default_constants())
    assert res.xyz.shape == (0, 3)
    assert res.intensity.shape == (0,)
    assert res.labels.shape == (0,)
