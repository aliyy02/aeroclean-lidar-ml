"""Publish labeled scans as colored PointCloud2 for RViz, to eyeball labeling quality.

Loads `data/labeled/<bag>/frame_*.npz`, colors each point by its class, converts NED->a
z-up view frame, and lays several bags side by side (a gallery) with the GT window
outlines drawn as red boxes. Open RViz with `calibration/labeled.rviz` (fixed frame `map`).

    source /opt/ros/jazzy/setup.bash
    python3 -m calibration.rviz_publish --bags 1,7,20,25,43

The pure helpers (label_to_rgb, ned_to_view) are ROS-free and unit-tested.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List

import numpy as np

from .label import NOT_GLASS, GLASS, GROUND, INTERIOR

# class -> RGB, high-contrast & distinct (no brown/orange confusion)
_COLORS = {
    NOT_GLASS: (255, 235, 0),     # yellow - frame / mullion
    GLASS:     (0, 140, 255),     # blue   - glass
    GROUND:    (0, 200, 70),      # green  - floor
    INTERIOR:  (255, 40, 40),     # red    - deep see-through interior
}


def label_to_rgb(labels: np.ndarray) -> np.ndarray:
    """Map (N,) class ids to (N,3) uint8 RGB."""
    labels = np.asarray(labels)
    rgb = np.full((labels.shape[0], 3), 255, dtype=np.uint8)
    for lid, c in _COLORS.items():
        rgb[labels == lid] = c
    return rgb


def ned_to_view(xyz_ned: np.ndarray) -> np.ndarray:
    """Body-NED (x fwd, y right, z down) -> view frame (X right, Y forward, Z up)."""
    xyz_ned = np.asarray(xyz_ned, dtype=float)
    out = np.empty_like(xyz_ned)
    out[:, 0] = xyz_ned[:, 1]      # right
    out[:, 1] = xyz_ned[:, 0]      # forward
    out[:, 2] = -xyz_ned[:, 2]     # up
    return out


def load_bag_labeled(out_dir: str, max_pts: int = 60000, seed: int = 0):
    """Aggregate all frames of a labeled bag -> (xyz_ned (M,3), label (M,)), downsampled."""
    files = sorted(glob.glob(os.path.join(out_dir, "frame_*.npz")))
    xyz, lab = [], []
    for f in files:
        d = np.load(f)
        xyz.append(d["xyz"]); lab.append(d["label"])
    xyz = np.concatenate(xyz); lab = np.concatenate(lab)
    if xyz.shape[0] > max_pts:
        idx = np.random.default_rng(seed).choice(xyz.shape[0], max_pts, replace=False)
        xyz, lab = xyz[idx], lab[idx]
    return xyz, lab


# ---- ROS-only below (imported lazily so the helpers stay importable without ROS) ----

def _make_xyzrgb(points_view: np.ndarray, rgb: np.ndarray, frame_id: str, stamp):
    from sensor_msgs.msg import PointCloud2, PointField
    n = points_view.shape[0]
    arr = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "u4")])
    arr["x"] = points_view[:, 0]; arr["y"] = points_view[:, 1]; arr["z"] = points_view[:, 2]
    arr["rgb"] = (rgb[:, 0].astype(np.uint32) << 16 |
                  rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2].astype(np.uint32))
    msg = PointCloud2()
    msg.header.frame_id = frame_id; msg.header.stamp = stamp
    msg.height = 1; msg.width = n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
    ]
    msg.is_bigendian = False; msg.point_step = 16; msg.row_step = 16 * n
    msg.is_dense = True; msg.data = arr.tobytes()
    return msg


def _window_markers(gt, offset_x: float, bag_name: str, idx: int, frame_id: str, stamp):
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point
    out = []
    for j, w in enumerate(gt.windows.values()):
        corners = [w["UL"], w["UR"], w["LR"], w["LL"], w["UL"]]
        view = ned_to_view(np.array(corners))
        m = Marker()
        m.header.frame_id = frame_id; m.header.stamp = stamp
        m.ns = f"{bag_name}_gt"; m.id = idx * 100 + j; m.type = Marker.LINE_STRIP
        m.action = Marker.ADD; m.scale.x = 0.02
        m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0
        m.points = [Point(x=float(p[0] + offset_x), y=float(p[1]), z=float(p[2])) for p in view]
        out.append(m)
    t = Marker()
    t.header.frame_id = frame_id; t.header.stamp = stamp
    t.ns = f"{bag_name}_label"; t.id = idx * 100 + 99; t.type = Marker.TEXT_VIEW_FACING
    t.action = Marker.ADD; t.scale.z = 0.25
    t.color.r = t.color.g = t.color.b = 1.0; t.color.a = 1.0
    t.text = bag_name
    top = ned_to_view(np.array([next(iter(gt.windows.values()))["UL"]]))[0]
    t.pose.position.x = float(top[0] + offset_x); t.pose.position.y = float(top[1])
    t.pose.position.z = float(top[2] + 0.4)
    out.append(t)
    return out


def main(argv=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from visualization_msgs.msg import MarkerArray
    from .gt_parse import load_gt

    ap = argparse.ArgumentParser(description="Publish labeled scans for RViz.")
    ap.add_argument("--bags", default="1,7,20,25,43", help="comma list of bag numbers")
    ap.add_argument("--data", default="data/labeled")
    ap.add_argument("--gt-dir", default="L6_tests_1/L6_tests_1")
    ap.add_argument("--name-fmt", default="L6_test_{n}", help="npz subdir name format")
    ap.add_argument("--gt-fmt", default="L6_test_{n}", help="GT file stem format (without .txt)")
    ap.add_argument("--max-pts", type=int, default=60000, help="max points per bag")
    ap.add_argument("--spacing", type=float, default=9.0, help="gallery spacing (m)")
    ap.add_argument("--rate", type=float, default=1.0)
    a = ap.parse_args(argv)
    nums = [int(x) for x in a.bags.split(",") if x.strip()]

    pts_all, rgb_all, markers = [], [], []
    for i, num in enumerate(nums):
        name = a.name_fmt.format(n=num)
        out_dir = os.path.join(a.data, name)
        if not os.path.isdir(out_dir):
            print(f"skip {name}: no labeled output at {out_dir}"); continue
        xyz, lab = load_bag_labeled(out_dir, a.max_pts)
        view = ned_to_view(xyz); view[:, 0] += i * a.spacing
        pts_all.append(view); rgb_all.append(label_to_rgb(lab))
        gt = load_gt(os.path.join(a.gt_dir, f"{a.gt_fmt.format(n=num)}.txt"))
        markers.append((gt, i * a.spacing, name, i))
        print(f"{name}: {xyz.shape[0]} pts")
    if not pts_all:
        print("nothing to publish"); return
    points = np.vstack(pts_all); rgb = np.vstack(rgb_all)

    import time
    rclpy.init(args=argv)
    node = Node("labeled_cloud_publisher")
    qos = QoSProfile(depth=1); qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    cloud_pub = node.create_publisher(PointCloud2, "/labeled/cloud", qos)
    mark_pub = node.create_publisher(MarkerArray, "/labeled/markers", qos)
    time.sleep(0.6)  # let discovery settle, then warn about stale duplicate publishers
    n_pub = node.count_publishers("/labeled/cloud")
    if n_pub > 1:
        node.get_logger().warn(
            f"*** {n_pub} publishers on /labeled/cloud! A stale one is still running and RViz "
            f"will FLICKER between old and new data. Kill them first:  pkill -f rviz_publish ***")

    def tick():
        stamp = node.get_clock().now().to_msg()
        cloud_pub.publish(_make_xyzrgb(points, rgb, "map", stamp))
        ma = MarkerArray()
        for gt, off, name, idx in markers:
            ma.markers.extend(_window_markers(gt, off, name, idx, "map", stamp))
        mark_pub.publish(ma)

    tick()
    node.create_timer(1.0 / a.rate, tick)
    print(f"publishing {points.shape[0]} pts on /labeled/cloud (frame 'map'). Ctrl-C to stop.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
