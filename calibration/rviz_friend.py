"""Publish a friend's labeled clouds (the `Labeled/` .npz format) for RViz.

Their format = one aggregated cloud per scan: `xyz (N,3)`, `intensity (N,)`,
`label (N,) uint8` (binary: 0 = frame/not-glass, 1 = glass), `pose (6,)`, `scan (str)`,
in a panel-aligned NED-like frame (x fwd, y right, z down), panel-only (no floor).

Reuses the view-frame map + cloud builder from `rviz_publish`, and publishes on the SAME
topic `/labeled/cloud`, so the existing `calibration/labeled.rviz` works unchanged.

    source /opt/ros/jazzy/setup.bash
    python3 -m calibration.rviz_friend --bags 1,20,43
    rviz2 -d calibration/labeled.rviz     # fixed frame: map
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .rviz_publish import ned_to_view, _make_xyzrgb

# friend's 2-class scheme; colors match our convention (glass blue, frame yellow)
_COLORS = {0: (255, 235, 0), 1: (0, 140, 255)}
_NAMES = {0: "frame", 1: "glass"}


def label_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    rgb = np.full((labels.shape[0], 3), 255, dtype=np.uint8)
    for lid, c in _COLORS.items():
        rgb[labels == lid] = c
    return rgb


def load_friend(path: str, max_pts: int = 100000, seed: int = 0):
    d = np.load(path, allow_pickle=True)
    xyz = d["xyz"].astype(float)
    lab = np.asarray(d["label"])
    if xyz.shape[0] > max_pts:
        idx = np.random.default_rng(seed).choice(xyz.shape[0], max_pts, replace=False)
        xyz, lab = xyz[idx], lab[idx]
    return xyz, lab


def main(argv=None):
    import time
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from sensor_msgs.msg import PointCloud2

    ap = argparse.ArgumentParser(description="Publish a friend's Labeled/ clouds for RViz.")
    ap.add_argument("--bags", default="1,20,43", help="comma list of bag numbers")
    ap.add_argument("--dir", default="Labeled")
    ap.add_argument("--max-pts", type=int, default=100000, help="max points per scan")
    ap.add_argument("--spacing", type=float, default=9.0, help="gallery spacing (m)")
    ap.add_argument("--rate", type=float, default=1.0)
    a = ap.parse_args(argv)
    nums = [int(x) for x in a.bags.split(",") if x.strip()]

    pts_all, rgb_all = [], []
    for i, num in enumerate(nums):
        p = os.path.join(a.dir, f"L6_test_{num}.npz")
        if not os.path.isfile(p):
            print(f"skip L6_test_{num}: no file at {p}"); continue
        xyz, lab = load_friend(p, a.max_pts)
        v = ned_to_view(xyz); v[:, 0] += i * a.spacing
        pts_all.append(v); rgb_all.append(label_to_rgb(lab))
        frac = "  ".join(f"{_NAMES.get(int(k), k)}={100*np.mean(lab == k):.0f}%"
                         for k in np.unique(lab))
        print(f"L6_test_{num}: {xyz.shape[0]} pts  {frac}")
    if not pts_all:
        print("nothing to publish"); return
    points = np.vstack(pts_all); rgb = np.vstack(rgb_all)

    rclpy.init(args=argv)
    node = Node("labeled_friend_publisher")
    qos = QoSProfile(depth=1); qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    pub = node.create_publisher(PointCloud2, "/labeled/cloud", qos)
    time.sleep(0.6)
    if node.count_publishers("/labeled/cloud") > 1:
        node.get_logger().warn("*** another publisher is on /labeled/cloud — RViz will FLICKER; "
                               "kill stale ones:  pkill -f rviz_ ***")

    def tick():
        pub.publish(_make_xyzrgb(points, rgb, "map", node.get_clock().now().to_msg()))

    tick()
    node.create_timer(1.0 / a.rate, tick)
    print(f"publishing {points.shape[0]} pts on /labeled/cloud (frame 'map'). "
          f"glass=blue, frame=yellow. Ctrl-C to stop.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
