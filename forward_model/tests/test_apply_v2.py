"""Revised orchestrator: per point -> reported intensity (range-free) + detection
(transmit x return-chance x noise-floor). Surviving points carry rho as intensity."""
import numpy as np

from forward_model.material_library import sample_material
from forward_model.apply_v2 import apply_scan, Constants


def points_at(theta_deg, R, N, normal=(-1.0, 0.0, 0.0)):
    th = np.radians(theta_deg)
    p = R * np.array([np.cos(th), np.sin(th), 0.0])     # incidence theta to normal [-1,0,0]
    return np.tile(p, (N, 1)), np.tile(normal, (N, 1))


def frac_kept(mat, theta_deg, R, N=3000, T=1e-4, seed=0):
    pts, nrm = points_at(theta_deg, R, N)
    res = apply_scan(pts, nrm, mat, Constants(C=1.0, T=T, sigma=0.0),
                     np.random.default_rng(seed))
    return res.xyz.shape[0] / N, res


def test_opaque_wall_returns_when_close():
    mat = sample_material(np.random.default_rng(0), "wall")
    f, res = frac_kept(mat, 20, 2.0)
    assert f > 0.95                                # opaque always returns above noise floor
    assert np.all(res.intensity > 0)
    assert np.all(res.labels == mat.train_label)


def test_clear_glass_sparse_off_normal_denser_near_normal():
    mat = sample_material(np.random.default_rng(1), "glass_clear")
    f_normal, _ = frac_kept(mat, 2, 2.0)
    f_oblique, _ = frac_kept(mat, 60, 2.0)
    assert f_oblique < 0.10                        # mostly transmits / specular off-normal
    assert f_normal > f_oblique                    # denser near head-on


def test_coated_lowe_glass_returns_at_all_angles():
    mat = sample_material(np.random.default_rng(2), "glass_lowE")
    f_oblique, _ = frac_kept(mat, 60, 2.0)
    assert f_oblique > 0.4                          # still returns at steep angle


def test_reported_intensity_is_range_independent():
    mat = sample_material(np.random.default_rng(3), "wall")
    _, near = frac_kept(mat, 20, 1.5)
    _, far = frac_kept(mat, 20, 3.0)
    assert abs(near.intensity.mean() - far.intensity.mean()) < 1e-6   # rho, no 1/R^n


def test_far_faint_dropped_by_noise_floor():
    mat = sample_material(np.random.default_rng(4), "glass_clear")   # faint
    f_far, _ = frac_kept(mat, 2, 60.0, T=0.05)      # far + high threshold
    assert f_far < 0.05


def test_reported_intensity_saturates_at_255():
    """Real L2 intensity clips at 255 (Bech glazing ~87% saturated, frame ~14%); the reported
    reflectivity must be clamped even when the modelled rho exceeds the ceiling."""
    from forward_model.material_library import Material
    bright = Material(name="x", a=200.0, g=0.5, s=50.0, b=200.0, m=0.10,
                      p_floor=1.0, cone=0.3, tau=0.0, n_index=1.5, is_glass=False, train_label=0)
    pts, nrm = points_at(0.5, 2.0, 500)
    res = apply_scan(pts, nrm, bright, Constants(C=1.0, T=1e-4, sigma=0.0),
                     np.random.default_rng(0))
    assert res.intensity.max() <= 255.0 + 1e-9      # clamped
    assert np.any(res.intensity >= 254.0)           # and it does reach the ceiling
