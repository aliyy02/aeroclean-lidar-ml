"""Live raw-capture loop: one building = one episode, orbit grid of poses.

Per capture it saves ONLY geometry + labels + normals (no intensity, nothing culled):
  episode_NNN/cap_MMMM.npz   xyz (N,3) f32, label (N,) i16, normal (N,3) f32, + sensor pose
  episode_NNN/building.json  every box (name, class, center, R, half)
  index.csv                  one row per capture across all episodes

The world->body rotation is solved PER CAPTURE from the scene's own large surfaces
(buildings.normals.estimate_M), so AirSim's attitude convention is never assumed.
Normals are exact per-point box-face normals (Option A), including protrusion reveals.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from forward_model import materials as M
from .build import sample_building
from .orbit import capture_poses
from .normals import estimate_M, per_point_normals
from .calibrate_spawn import load_asset_size

MOUNT = np.array([1.0, 0.0, -0.05])      # lidar offset in vehicle frame (settings X,Y,Z)
MAX_RANGE = 30.0
# simSpawnObject scale is a MULTIPLIER on the project's "Cube" mesh (~2.5 m, not 1 m), so a
# requested scale comes out ~2.5x too big. Divide by the measured asset size so boxes spawn at
# their intended metre dimensions. Re-measure with `python3 -m buildings.calibrate_spawn`.
ASSET_SIZE = load_asset_size()


def _des_to_ned(p):
    o = np.empty_like(p); o[:, 0] = -p[:, 2]; o[:, 1] = p[:, 1]; o[:, 2] = p[:, 0]; return o


def _q_euler(airsim, roll, pitch, yaw):
    r, p, y = map(math.radians, (roll, pitch, yaw))
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return airsim.Quaternionr(sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                              cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


# Cosmetic materials (visual only -- labels come from the object name, not the material).
_MATERIAL = {
    M.GLASS_CLEAR: "/Game/M_Glass", M.GLASS_COATED: "/Game/M_Glass", M.GLASS_LOWE: "/Game/M_Glass",
    M.METAL_FRAME: "/Game/M_Aluminum", M.SPANDREL: "/Game/M_Concrete",
    M.WALL: "/Game/M_Concrete", M.GROUND: "/Game/M_Concrete",
}


def spawn_building(client, airsim, building, apply_materials=True):
    """Spawn every box. Square buildings are world-axis-aligned -> identity quaternion."""
    names = []
    for b in building.boxes:
        world_half = np.abs(b.R) @ b.half
        scale = 2 * world_half / ASSET_SIZE          # correct for the ~2.5 m "Cube" asset
        try:
            nm = client.simSpawnObject(b.name, "Cube",
                                       airsim.Pose(airsim.Vector3r(*b.center), airsim.Quaternionr(0, 0, 0, 1)),
                                       airsim.Vector3r(*scale), False, False)
            if nm:
                names.append(nm)
                if apply_materials and b.cls in _MATERIAL:
                    try:
                        client.simSetObjectMaterial(nm, _MATERIAL[b.cls])
                    except Exception:
                        pass
        except Exception as e:
            print(f"  ! spawn {b.name}: {e}")
    return names


def destroy(client, names):
    for nm in names:
        try:
            client.simDestroyObject(nm)
        except Exception:
            pass


def save_building_json(out: Path, building):
    boxes = [{"name": b.name, "cls": int(b.cls), "center": b.center.tolist(),
              "R": b.R.tolist(), "half": b.half.tolist()} for b in building.boxes]
    out.write_text(json.dumps({"params": building.params.__dict__, "boxes": boxes}, indent=1))


def run_episode(client, airsim, building, ep_dir: Path, rng, k=4, budget=None, max_caps=None, settle=0.2):
    budget = budget if budget is not None else max_caps     # max_caps kept as a budget alias
    ep_dir.mkdir(parents=True, exist_ok=True)
    index = {b.name: b for b in building.boxes}
    save_building_json(ep_dir / "building.json", building)
    spawned = spawn_building(client, airsim, building)
    rows = []
    try:
        for ci, pose in enumerate(capture_poses(building, rng, k=k, budget=budget)):
            # Place the VEHICLE so the lidar (mounted MOUNT[0] m forward, facing the wall)
            # lands at the intended standoff -- i.e. pull the vehicle back along the outward normal.
            veh_pos = pose.sensor_pos + MOUNT[0] * pose.outward
            client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(*veh_pos), _q_euler(airsim, pose.roll, pose.pitch, pose.yaw)),
                True, "CV1")
            import time; time.sleep(settle)
            vp = client.simGetVehiclePose("CV1")
            veh = np.array([vp.position.x_val, vp.position.y_val, vp.position.z_val])
            d = client.getLidarData(lidar_name="UnitreeL2Lidar", vehicle_name="CV1")
            if len(d.point_cloud) < 30:
                continue
            pts = np.asarray(d.point_cloud, np.float32).reshape(-1, 3)
            gt = list(getattr(d, "groundtruth", []) or [])
            dist = np.linalg.norm(pts, axis=1)
            keep = (dist > 0.1) & (dist < MAX_RANGE)
            pts = pts[keep]; names = [gt[j] for j in np.nonzero(keep)[0]]
            keep2 = np.array([nm in index for nm in names])
            if keep2.sum() < 100:
                continue
            body = _des_to_ned(pts[keep2]); names = [n for n, k2 in zip(names, keep2) if k2]
            Mwb, nsurf = estimate_M(body, names, index)
            if Mwb is None:
                continue
            sensor_world = veh + Mwb.T @ MOUNT
            normals = per_point_normals(body, names, index, Mwb, sensor_world)
            labels = np.array([M.material_for(n) for n in names], dtype=np.int16)
            cap = ep_dir / f"cap_{ci:04d}.npz"
            np.savez(cap, xyz=body.astype(np.float32), label=labels, normal=normals.astype(np.float32),
                     sensor_pos=sensor_world.astype(np.float32))
            rows.append({"cap": str(cap), "n": int(body.shape[0]), "face": pose.face_idx,
                         "standoff": round(pose.standoff, 2), "surfaces": nsurf})
    finally:
        destroy(client, spawned)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Raw building-orbit capture (no postprocessing).")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--region", default="gcc")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--k", type=int, default=4, help="shots per grid cell")
    ap.add_argument("--per-building", type=int, default=60,
                    help="target captures/building, drawn spread across all faces (0 = full grid)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-caps", type=int, default=None, help="hard cap/episode (debug); alias of --per-building")
    ap.add_argument("--square-prob", type=float, default=0.6)
    ap.add_argument("--host", default="172.28.144.1")
    args = ap.parse_args()

    import cosysairsim as airsim
    client = airsim.VehicleClient(ip=args.host); client.confirmConnection()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    budget = args.per_building if args.per_building and args.per_building > 0 else None
    all_rows = []
    for ep in range(args.episodes):
        building = sample_building(rng, region=args.region, square_prob=args.square_prob)
        ep_dir = out / f"episode_{ep:03d}"
        print(f"episode {ep}: {len(building.boxes)} boxes, {building.params.n_floors} floors, "
              f"{building.footprint.kind}, budget={budget}")
        rows = run_episode(client, airsim, building, ep_dir, rng, k=args.k,
                           budget=budget, max_caps=args.max_caps)
        for r in rows:
            r["episode"] = ep
        all_rows += rows
        print(f"  -> {len(rows)} captures")
    if all_rows:
        with open(out / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["episode", "cap", "n", "face", "standoff", "surfaces"])
            w.writeheader(); w.writerows(all_rows)
    print(f"Done: {len(all_rows)} captures -> {out}")


if __name__ == "__main__":
    main()
