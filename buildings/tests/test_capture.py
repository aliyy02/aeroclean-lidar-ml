"""Offline end-to-end test of the live capture loop (buildings/capture.py run_episode).

The sim is replaced by a FAKE client whose getLidarData() synthesises a point cloud from the
building's own box faces: it picks the faces pointing at the sensor, samples their front planes
in world coordinates, applies a KNOWN world->body rotation M_TRUE, and returns them in the raw
lidar frame (the inverse of capture's des_to_ned). So the test exercises the real pipeline --
range filter, des_to_ned, the Kabsch rotation solve, per-point normals, labelling, and the .npz
save -- and checks the saved data is correct, all without UE5.
"""
import math
from types import SimpleNamespace

import numpy as np

from forward_model import materials as M
from buildings.footprint import footprint_from_vertices
from buildings.build import build_building, sample_building_params
from buildings.capture import run_episode, MOUNT

GLASS = {M.GLASS_CLEAR, M.GLASS_COATED, M.GLASS_LOWE}
MAX_RANGE = 30.0


def _rot(axis, deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    if axis == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# An arbitrary but fixed valid rotation (mimics the -90 deg lidar mount + some yaw/roll).
M_TRUE = _rot("z", 35) @ _rot("y", -90) @ _rot("x", 8)
_NED_FROM_DES = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])     # body = D @ lidar  (== des_to_ned)


def _ned_to_des(body):                                          # lidar = D^T @ body
    return _NED_FROM_DES.T @ body


def _sample_front(box, step=0.25, cap=400):
    """World points on a box's +n front face (a grid no finer than `step`, capped)."""
    na = max(2, min(int(2 * box.half[1] / step) + 1, 25))
    nu = max(2, min(int(2 * box.half[2] / step) + 1, 25))
    us = np.linspace(-box.half[1], box.half[1], na)
    vs = np.linspace(-box.half[2], box.half[2], nu)
    front = box.center + box.half[0] * box.R[:, 0]
    pts = [front + a * box.R[:, 1] + b * box.R[:, 2] for a in us for b in vs]
    return pts[:cap]


class _FakeClient:
    """Minimal stand-in for the CosysAirSim VehicleClient used by run_episode/spawn_building."""

    def __init__(self, building):
        self.building = building
        self.veh = np.zeros(3)

    # spawn/cleanup are no-ops that just echo the name
    def confirmConnection(self): pass
    def simListSceneObjects(self): return []
    def simSpawnObject(self, name, asset, pose, scale, *a): return name
    def simDestroyObject(self, name): pass
    def simSetObjectMaterial(self, name, material): pass

    def simSetVehiclePose(self, pose, ignore_collision, vehicle):
        self.veh = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])

    def simGetVehiclePose(self, vehicle):
        return SimpleNamespace(position=SimpleNamespace(
            x_val=self.veh[0], y_val=self.veh[1], z_val=self.veh[2]))

    def getLidarData(self, lidar_name, vehicle_name):
        sensor = self.veh + M_TRUE.T @ MOUNT                    # true lidar world position
        cloud, names = [], []
        for b in self.building.boxes:
            if b.cls == M.GROUND:                               # sample a patch of floor near the sensor
                gx, gy = sensor[0], sensor[1]
                for ax in np.linspace(gx - 6, gx + 6, 18):
                    for ay in np.linspace(gy - 6, gy + 6, 18):
                        cloud.append(np.array([ax, ay, -0.02])); names.append(b.name)
                continue
            if (sensor - b.center) @ b.R[:, 0] <= 0:            # front face must point at the sensor
                continue
            for wp in _sample_front(b):
                if np.linalg.norm(wp - sensor) < MAX_RANGE:
                    cloud.append(wp); names.append(b.name)
        lidar = [_ned_to_des(M_TRUE @ (np.asarray(wp) - sensor)) for wp in cloud]
        flat = np.asarray(lidar, np.float32).reshape(-1).tolist() if lidar else []
        return SimpleNamespace(point_cloud=flat, groundtruth=names, time_stamp=0)


def _build():
    rng = np.random.default_rng(1)
    fp = footprint_from_vertices([(-8, -7), (8, -7), (8, 7), (-8, 7)], kind="square")
    return build_building(fp, sample_building_params(rng, "gcc"), rng)


def test_run_episode_saves_correct_labeled_cloud(tmp_path):
    bld = _build()
    rng = np.random.default_rng(0)
    rows = run_episode(_FakeClient(bld), SimpleNamespace(
        Vector3r=lambda x, y, z: SimpleNamespace(x_val=x, y_val=y, z_val=z),
        Quaternionr=lambda x, y, z, w: SimpleNamespace(x_val=x, y_val=y, z_val=z, w_val=w),
        Pose=lambda position, orientation: SimpleNamespace(position=position, orientation=orientation)),
        bld, tmp_path, rng, k=2, max_caps=3, settle=0.0)

    assert rows, "no captures were saved"
    caps = sorted(tmp_path.glob("cap_*.npz"))
    assert caps, "no .npz written"
    d = np.load(caps[0])
    assert set(d.files) >= {"xyz", "label", "normal", "sensor_pos"}
    assert d["xyz"].shape[0] > 100 and d["xyz"].shape == d["normal"].shape
    # labels must be real material ids that include frame, some glass, and ground
    labs = set(int(x) for x in d["label"])
    assert M.METAL_FRAME in labs and (labs & GLASS) and M.GROUND in labs
    # every per-point normal is unit length and oriented back toward the sensor (origin in body)
    nrm = d["normal"]
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5)
    assert np.all((d["xyz"] * nrm).sum(1) <= 1e-6)
    # the Kabsch solve recovered M_TRUE end-to-end: mapping each body normal back by M_TRUE.T must
    # yield an axis-aligned world normal (this square building's faces are all +/-X, +/-Y, or up).
    world_n = nrm @ M_TRUE                            # row-wise M_TRUE.T @ n
    assert np.allclose(np.abs(world_n).max(1), 1.0, atol=1e-3)
    assert np.allclose(np.abs(world_n).sum(1), 1.0, atol=1e-3)
