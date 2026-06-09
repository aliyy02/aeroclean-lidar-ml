#!/usr/bin/env python3
"""Spawn a building and publish the LABELED Standard-Lidar cloud for RViz validation.

This is the honest way to check the data: it publishes exactly what the model will see
-- per-point (x, y, z, label) where label is the 8-class material id -- so colouring by
`label` in RViz shows the segmentation directly. Drive the ComputerVision camera around
in UE5 and the labeled cloud updates live.

Run (CV mode active = settings_cv_stdlidar.json, UE5 in Play):
    source /opt/ros/jazzy/setup.bash
    cd ~/ros2_aeroclean_ws/ml_pipeline
    python3 -m buildings.ros_publish --seed 2 --region gcc --square-prob 1.0

RViz:  Fixed Frame = unilidar_lidar ; add PointCloud2 on /lidar/cloud ;
       Color Transformer = Intensity, Channel Name = label  (one colour per class).
Class ids: 0 wall, 1 frame, 2 glass_clear, 3 glass_coated, 4 glass_lowE, 5 spandrel,
           6 ground, 7 other.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from forward_model import materials as M
from buildings.build import sample_building
from buildings.capture import spawn_building, destroy, _des_to_ned

HOST = os.environ.get("COSYS_IP", "172.28.144.1")
VEHICLE, SENSOR, FRAME = "CV1", "UnitreeL2Lidar", "unilidar_lidar"
MAX_RANGE = 40.0


def _ned_to_flu(p):
    o = np.empty_like(p)
    o[:, 0] = p[:, 0]; o[:, 1] = -p[:, 1]; o[:, 2] = -p[:, 2]
    return o


# Fixed colour per class so RViz (Color Transformer = RGB8) shows the segmentation
# unambiguously -- no rainbow/autocompute guesswork.
_COLORS = {0: (150, 150, 150), 1: (220, 40, 40), 2: (40, 200, 230), 3: (40, 90, 230),
           4: (170, 70, 230), 5: (220, 130, 40), 6: (40, 180, 60), 7: (235, 235, 235)}
COLOR_LEGEND = "  ".join(f"{M.CLASS_NAMES[k]}={_COLORS[k]}" for k in sorted(_COLORS))


def _pack_rgb(label):
    rgb = np.zeros(label.shape[0], np.uint32)
    for k, (r, g, b) in _COLORS.items():
        rgb[label == k] = (int(r) << 16) | (int(g) << 8) | int(b)
    return rgb.view(np.float32)


def _cloud(xyz, label, stamp):
    n = xyz.shape[0]
    m = PointCloud2()
    m.header.stamp = stamp; m.header.frame_id = FRAME
    m.height = 1; m.width = n; m.point_step = 20; m.row_step = 20 * n
    m.is_dense = True; m.is_bigendian = False
    m.fields = [PointField(name=k, offset=o, datatype=PointField.FLOAT32, count=1)
                for k, o in (("x", 0), ("y", 4), ("z", 8), ("label", 12), ("rgb", 16))]
    if n:
        flat = np.empty(n * 5, np.float32)
        flat[0::5] = xyz[:, 0]; flat[1::5] = xyz[:, 1]; flat[2::5] = xyz[:, 2]
        flat[3::5] = label; flat[4::5] = _pack_rgb(label.astype(int))
        m.data = flat.tobytes()
    else:
        m.data = b""
    return m


class Pub(Node):
    def __init__(self, client, index):
        super().__init__("building_lidar_pub")
        self.c = client; self.index = index
        self.pub = self.create_publisher(PointCloud2, "/lidar/cloud", 10)
        self.tf = TransformBroadcaster(self)
        self.create_timer(0.15, self.tick)          # ~6-7 Hz
        self.n = 0
        self.get_logger().info("Publishing /lidar/cloud. RViz: Fixed Frame=unilidar_lidar, "
                               "Color Transformer=RGB8 (fixed colours per class).")

    def tick(self):
        d = self.c.getLidarData(lidar_name=SENSOR, vehicle_name=VEHICLE)
        if len(d.point_cloud) < 3:
            return
        pts = np.asarray(d.point_cloud, np.float32).reshape(-1, 3)
        gt = list(getattr(d, "groundtruth", []) or [])
        dist = np.linalg.norm(pts, axis=1)
        keep = (dist > 0.1) & (dist < MAX_RANGE)
        pts = pts[keep]; names = [gt[j] for j in np.nonzero(keep)[0]]
        if not names:
            return
        xyz = _ned_to_flu(_des_to_ned(pts))
        labels = np.array([M.material_for(n) for n in names], dtype=np.float32)
        stamp = self.get_clock().now().to_msg()
        self.pub.publish(_cloud(xyz, labels, stamp))

        t = TransformStamped(); t.header.stamp = stamp
        t.header.frame_id = "world"; t.child_frame_id = FRAME
        t.transform.rotation.w = 1.0
        self.tf.sendTransform(t)

        self.n += 1
        if self.n % 20 == 0:
            from collections import Counter
            cc = Counter(int(l) for l in labels)
            self.get_logger().info("pts=%d  " % len(labels) +
                                   " ".join(f"{M.CLASS_NAMES[k]}={v}" for k, v in sorted(cc.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--region", default="gcc")
    ap.add_argument("--square-prob", type=float, default=1.0)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--no-materials", action="store_true")
    args = ap.parse_args()

    import cosysairsim as airsim
    c = airsim.VehicleClient(ip=args.host); c.confirmConnection()
    print("clearing old actors...")
    for o in list(c.simListSceneObjects()):
        if o.startswith(("Frame_", "Glass", "Spandrel_", "Wall_", "Ground_", "Diag", "Ref", "Calib")):
            try:
                c.simDestroyObject(o)
            except Exception:
                pass
    time.sleep(0.6)
    bld = sample_building(np.random.default_rng(args.seed), region=args.region, square_prob=args.square_prob)
    print(f"spawning system {bld.params.system} building, {bld.params.n_floors} floors, {len(bld.boxes)} boxes...")
    names = spawn_building(c, airsim, bld, apply_materials=not args.no_materials)
    index = {b.name: b for b in bld.boxes}
    print(f"spawned {len(names)} boxes. Move the CV camera in UE5 to drive around.")
    print("RViz: set the PointCloud2 Color Transformer to RGB8. Class colours (R,G,B):")
    print("  " + COLOR_LEGEND)

    rclpy.init()
    node = Pub(c, index)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        destroy(c, names)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
