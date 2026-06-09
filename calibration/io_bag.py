"""Read PointCloud2 frames from a rosbag2 mcap (native L2 frame, meters).

This is the one ROS-dependent module (uses rosbag2_py + sensor_msgs_py); source the
ROS 2 environment before calling. Pure logic lives elsewhere and is tested offline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import numpy as np


@dataclass
class Frame:
    stamp_ns: int
    xyz: np.ndarray         # (N,3) native L2, meters
    intensity: np.ndarray   # (N,)


def read_frames(bag_path: str, topic: str = "/unilidar/cloud",
                drop_zero: bool = True) -> Iterator[Frame]:
    """Yield each PointCloud2 message on `topic` as a Frame. Drops all-zero fill points."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2 as pc2

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                            output_serialization_format="cdr"))
    while reader.has_next():
        tname, data, t = reader.read_next()
        if tname != topic:
            continue
        msg = deserialize_message(data, PointCloud2)
        arr = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"),
                              skip_nans=False)
        try:
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float64)
            inten = np.asarray(arr["intensity"], np.float64)
        except (IndexError, ValueError, TypeError):
            a = np.array([[p[0], p[1], p[2], p[3]] for p in arr], dtype=np.float64)
            xyz, inten = a[:, :3], a[:, 3]
        stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if drop_zero:
            keep = ~np.all(xyz == 0.0, axis=1)
            xyz, inten = xyz[keep], inten[keep]
        yield Frame(stamp_ns=stamp, xyz=xyz, intensity=inten)


def read_all(bag_path: str, topic: str = "/unilidar/cloud") -> List[Frame]:
    """Eagerly read all frames into a list."""
    return list(read_frames(bag_path, topic))
