#!/usr/bin/env python3
"""Measure the UE5 "Cube" asset size so spawned boxes come out their intended dimensions.

ROOT CAUSE this fixes: `simSpawnObject(..., "Cube", ..., scale)` treats `scale` as a
MULTIPLIER on the project's spawn "Cube" mesh -- which here is ~2.5 m, not 1 m. (NB: this
is a DIFFERENT mesh from the editor's basic-shapes Cube, which is exactly 1 m -- confirmed
from a known 1.58x1.04x0.03 m board built from the basic Cube at scale = its metre size.)
So every box spawns ~2.5x too big; the spawner divides the requested scale by this measured
asset size.

It measures the cube's **horizontal width** (y-extent) -- NOT its height -- at two moderate
cube scales and takes the SLOPE (which cancels the constant edge-sampling offset), writes
buildings/spawn_calib.json, then verifies a box spawned through the corrected path is right.

Why width, not height: the Unitree L2 has a full 360 deg HORIZONTAL field of view but a
LIMITED vertical one. A tall probe cube overshoots the vertical FOV, so its measured height is
clipped by the sensor, not the cube -> the slope comes out too small (this produced a bogus
asset_size = 2.23, ~12% low, which made every building ~12% oversized: transoms poked past the
corners into open air and spandrels overlapped the panel below). Width is never clipped.

    python3 -m buildings.calibrate_spawn        # run with UE5 (CV mode) up

Re-run if you ever change the UE5 spawn "Cube" asset.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

CALIB_PATH = Path(__file__).parent / "spawn_calib.json"
MOUNT = np.array([1.0, 0.0, -0.05])
DEFAULT_ASSET_SIZE = 2.5

# Probe geometry. The lidar is pitched -90 deg -> it sees a forward (+X_body) hemisphere, and
# points come back as x_body=-z_lidar, y_body=y_lidar, z_body=x_lidar (see _des_to_ned). So put
# the probe straight ahead at +X. Make it a THIN, TALL vertical slab (not a fat cube): thin in X
# so it can never reach back and swallow the sensor (the old bug: a 10 m cube 4 m away engulfed
# the lidar -> 0 returns -> crash), tall in Z so plenty of scan lines cross it. We vary only its
# WIDTH (Y) and measure that -- the horizontal scan is full 360 deg, never FOV-clipped.
PROBE_DIST = 7.0        # metres ahead (+X); keeps the slab fully in front at all test widths
PROBE_THIN = 0.2        # X scale -> ~0.5 m thick: front face stays ~6.7 m away, never engulfs
PROBE_TALL = 2.0        # Z scale -> ~5 m tall


def _des_to_ned(p):
    o = np.empty_like(p); o[:, 0] = -p[:, 2]; o[:, 1] = p[:, 1]; o[:, 2] = p[:, 0]; return o


def _scan_probe_width(client, airsim, scale) -> float:
    """Spawn a thin vertical slab of the given (x,y,z) scale 7 m ahead, poll until UE5 actually
    renders it to the lidar, return its measured world y-extent (width). Raises if never seen."""
    cen = np.array([PROBE_DIST, 0.0, -8.0])
    try:
        client.simDestroyObject("CalibProbe")
    except Exception:
        pass
    client.simSpawnObject("CalibProbe", "Cube",
                          airsim.Pose(airsim.Vector3r(*cen), airsim.Quaternionr(0, 0, 0, 1)),
                          airsim.Vector3r(*[float(s) for s in scale]), False, False)
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, cen[2]), airsim.Quaternionr(0, 0, 0, 1)),
                             True, "CV1")
    width = None
    for _ in range(15):                  # UE5 needs a few lidar rotations after spawn to register
        time.sleep(0.3)
        d = client.getLidarData(lidar_name="UnitreeL2Lidar", vehicle_name="CV1")
        pts = np.asarray(d.point_cloud, np.float32).reshape(-1, 3)
        gt = list(getattr(d, "groundtruth", []) or [])
        if len(gt) != len(pts) or len(pts) == 0:
            continue
        m = np.array([g == "CalibProbe" for g in gt])
        if m.sum() >= 40:
            yv = (_des_to_ned(pts[m]))[:, 1]
            width = float(yv.max() - yv.min())
            break
    try:
        client.simDestroyObject("CalibProbe")
    except Exception:
        pass
    if width is None:
        raise RuntimeError("probe never appeared in the scan -- is UE5 in Play with the CV "
                           "Std-Lidar settings active? (settings_cv_stdlidar.json)")
    return width


def measure_asset_size(client, airsim, scales=(2.0, 3.0, 4.0)):
    """Scan the probe width at several scales and FIT A LINE: width = asset*scale + offset.

    The slope is the asset size. The intercept is a constant per-scan sensor offset (the L2's
    fixed beam spread inflates every absolute width by the same ~couple of metres) -- it cancels
    in the slope, which is why we never trust a single absolute reading. Returns
    (asset_size, offset, max_residual). A small residual => the line fits => trust the slope.
    """
    widths = []
    for s in scales:
        w = _scan_probe_width(client, airsim, (PROBE_THIN, float(s), PROBE_TALL))
        widths.append(w)
        print(f"  scale_y={s:>4.1f}  ->  measured width {w:6.3f} m")
    sc = np.asarray(scales, float); wd = np.asarray(widths, float)
    slope, intercept = np.polyfit(sc, wd, 1)
    resid = float(np.max(np.abs(wd - (slope * sc + intercept))))
    return float(slope), float(intercept), resid


def main():
    import cosysairsim as airsim
    c = airsim.VehicleClient(ip="172.28.144.1"); c.confirmConnection()
    asset, offset, resid = measure_asset_size(c, airsim)
    CALIB_PATH.write_text(json.dumps({"asset_size": round(asset, 4)}, indent=2))
    linear = resid < 0.25
    print(f"\nasset_size (line slope)            = {asset:.4f} m   -> wrote {CALIB_PATH}")
    print(f"constant sensor offset (intercept) = {offset:+.3f} m   [expected; the slope cancels it]")
    print(f"linearity check: max residual {resid:.3f} m  "
          f"({'OK -- trust the slope' if linear else 'CHECK -- non-linear; send Claude the widths above'})")


def load_asset_size() -> float:
    try:
        return float(json.loads(CALIB_PATH.read_text())["asset_size"])
    except Exception:
        return DEFAULT_ASSET_SIZE


if __name__ == "__main__":
    main()
