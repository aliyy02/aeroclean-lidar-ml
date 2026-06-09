"""Write the labeled per-frame .npz back out as ROS2 (mcap) bags, with labels.

Each labeled scan dir (data/labeled_<ds>/<scan>/frame_*.npz + index.csv) becomes a
rosbag2 mcap with a PointCloud2 on /labeled/cloud carrying fields
x,y,z,intensity,label (label as float32 so RViz can colour by it). Points are in the
corrected-NED frame (axis-mapped + de-rolled, == GT frame), frame_id='lidar'.

Usage (ROS sourced):
  PYTHONPATH=.:$PYTHONPATH python3 analysis/labeled_to_bag.py <labeled_dir> <out_dir> [one]
  e.g. python3 analysis/labeled_to_bag.py data/labeled_oxy data/labeled_bags/oxy
"""
import sys
import os
import csv
import glob
import numpy as np


def write_scan_bag(scan_dir, out_dir):
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs_py import point_cloud2 as pc2
    from std_msgs.msg import Header

    rows = list(csv.DictReader(open(os.path.join(scan_dir, "index.csv"))))
    if not rows:
        return 0
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=out_dir, storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    topic = rosbag2_py.TopicMetadata(0, "/labeled/cloud",
                                     "sensor_msgs/msg/PointCloud2", "cdr")
    writer.create_topic(topic)
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
              for i, n in enumerate(("x", "y", "z", "intensity", "label"))]
    n_written = 0
    for row in rows:
        d = np.load(os.path.join(scan_dir, row["file"]))
        xyz = d["xyz"].astype(np.float32)
        pts = np.column_stack([xyz, d["intensity"].astype(np.float32),
                               d["label"].astype(np.float32)])
        stamp_ns = int(row["stamp_ns"])
        hdr = Header()
        hdr.stamp.sec = stamp_ns // 1_000_000_000
        hdr.stamp.nanosec = stamp_ns % 1_000_000_000
        hdr.frame_id = "lidar"
        msg = pc2.create_cloud(hdr, fields, pts)
        writer.write("/labeled/cloud", serialize_message(msg), stamp_ns)
        n_written += 1
    del writer
    return n_written


def main():
    labeled_dir, out_root = sys.argv[1], sys.argv[2]
    only_one = len(sys.argv) > 3 and sys.argv[3] == "one"
    scans = sorted([d for d in glob.glob(os.path.join(labeled_dir, "*"))
                    if os.path.isfile(os.path.join(d, "index.csv"))])
    if only_one:
        scans = scans[:1]
    os.makedirs(out_root, exist_ok=True)
    total = 0
    for s in scans:
        name = os.path.basename(s)
        nf = write_scan_bag(s, os.path.join(out_root, name))
        total += nf
        print(f"  {name}: {nf} frames -> {os.path.join(out_root, name)}")
    print(f"DONE {labeled_dir}: {len(scans)} bags, {total} frames")


if __name__ == "__main__":
    main()
