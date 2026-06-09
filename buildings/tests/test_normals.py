"""Tests for per-point face normals + the per-capture world->body rotation solver."""
import numpy as np

from forward_model import materials as M
from buildings.build import Box
from buildings.normals import (
    face_normal_world, sensor_facing_world_normal,
    solve_world_to_body, estimate_M, per_point_normals,
)


def mullion():
    # vertical protruding frame bar: thin in X (n), narrow in Y (t), tall in Z (u)
    return Box(name="Frame_1", cls=M.METAL_FRAME, center=np.array([3.0, 0.0, 0.0]),
               R=np.eye(3), half=np.array([0.02, 0.03, 1.0]))


def test_front_face_point_gets_outward_normal():
    n = face_normal_world(np.array([3.02, 0.0, 0.4]), mullion())   # on +X front face
    np.testing.assert_allclose(n, [1, 0, 0], atol=1e-9)


def test_protrusion_reveal_point_gets_perpendicular_normal():
    n = face_normal_world(np.array([3.0, 0.03, 0.4]), mullion())   # on +Y reveal
    np.testing.assert_allclose(n, [0, 1, 0], atol=1e-9)
    assert abs(n @ np.array([1, 0, 0])) < 1e-9                     # perpendicular to glass normal


def test_sensor_facing_normal_outward_for_wall_up_for_ground():
    wall = Box("Wall_1", M.WALL, np.array([3.0, 0, -5]), np.eye(3), np.array([0.05, 2, 2]))
    np.testing.assert_allclose(sensor_facing_world_normal(wall), [1, 0, 0])
    ground = Box("Ground_Plane", M.GROUND, np.zeros(3), np.eye(3), np.array([20, 20, 0.02]))
    np.testing.assert_allclose(sensor_facing_world_normal(ground), [0, 0, -1])


def test_solve_world_to_body_recovers_a_known_rotation():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3)); Q, _ = np.linalg.qr(A)            # random rotation
    Mtrue = Q * np.sign(np.linalg.det(Q))
    wn = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    bn = (Mtrue @ wn.T).T
    Mest = solve_world_to_body(wn, bn)
    assert np.linalg.norm(Mest - Mtrue) < 1e-6


# wall whose outward (R[:,0]) faces -X, toward a sensor sitting at lower X
_RZ180 = np.array([[-1., 0, 0], [0, -1., 0], [0, 0, 1.]])


def test_estimate_M_from_a_synthetic_two_surface_scan():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3)); Q, _ = np.linalg.qr(A); Mtrue = Q * np.sign(np.linalg.det(Q))
    sensor = np.array([0.0, 0.0, -5.0])
    wall = Box("Wall_1", M.WALL, np.array([4, 0, -5]), _RZ180, np.array([0.05, 3, 3]))
    ground = Box("Ground_Plane", M.GROUND, np.array([2, 0, 0]), np.eye(3), np.array([20, 20, 0.02]))
    index = {"Wall_1": wall, "Ground_Plane": ground}
    pts = []; names = []
    g = np.array(np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-6, -4, 12))).reshape(2, -1).T
    for yy, zz in g:                                # wall front face toward sensor at x=3.95
        pts.append([3.95, yy, zz]); names.append("Wall_1")
    for xx, yy in g:                                # ground top face at z=-0.02 (up)
        pts.append([xx + 2, yy, -0.02]); names.append("Ground_Plane")
    body = (Mtrue @ (np.array(pts) - sensor).T).T
    Mest, nsurf = estimate_M(body, names, index, sensor_world=sensor)
    assert nsurf >= 2
    assert np.linalg.norm(Mest - Mtrue) < 1e-3


def test_estimate_M_underdetermined_returns_none():
    wall = Box("Wall_1", M.WALL, np.array([4, 0, -5]), _RZ180, np.array([0.05, 3, 3]))
    index = {"Wall_1": wall}
    g = np.array(np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-6, -4, 10))).reshape(2, -1).T
    body = np.array([[3.95, yy, zz] for yy, zz in g]); names = ["Wall_1"] * len(body)
    Mest, nsurf = estimate_M(body, names, index, sensor_world=np.zeros(3))
    assert Mest is None                                            # one plane -> can't fix rotation


def test_per_point_normals_pick_front_axis_and_point_toward_sensor():
    box = mullion(); index = {"Frame_1": box}
    sensor = np.array([10.0, 0, 0])                                # +X side -> sees the +X front face
    world = np.array([[3.02, 0.0, 0.4], [3.02, 0.01, -0.3]])       # on the front face
    body = world - sensor                                          # world = sensor + I@body
    nb = per_point_normals(body, ["Frame_1", "Frame_1"], index, np.eye(3), sensor)
    for i in range(2):
        np.testing.assert_allclose(np.abs(nb[i]), [1, 0, 0], atol=1e-6)   # front axis
        assert body[i] @ nb[i] < 0                                 # normal points toward the sensor


def test_per_point_normals_orient_flips_a_back_facing_normal():
    # If geometry/M error yields an away-pointing normal, it is flipped toward the sensor.
    box = mullion(); index = {"Frame_1": box}
    sensor = np.array([10.0, 0, 0])
    body = np.array([[3.02, 0.0, 0.4]]) - sensor
    nb = per_point_normals(body, ["Frame_1"], index, np.eye(3), sensor)
    assert nb[0] @ (sensor - (sensor + body[0])) > 0              # points back toward the sensor
