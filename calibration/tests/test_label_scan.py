"""Orchestration: a list of native frames -> per-frame labeled clouds (kept points)."""
import os

import numpy as np

from calibration.io_bag import Frame
from calibration.gt_parse import load_gt
from calibration.label_scan import process_frames
from calibration.label import INTERIOR
from forward_model import materials as M

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "fixtures", "real_frames.npz")
GT = os.path.join(HERE, "fixtures", "L6_gt.txt")


def load_frames():
    d = np.load(FIX)
    return [Frame(int(d["stamps"][k]), d[f"f{k}_xyz"].astype(float),
                  d[f"f{k}_int"].astype(float)) for k in range(3)]


def test_glass_dominates_interior_after_panel_offset_fix():
    # The glass pane sits ~16 cm behind the GT plane (accumulated measurement offset).
    # process_frames auto-estimates that offset, so the recessed pane is GLASS (the bulk)
    # and only the sparse deep see-through tail is INTERIOR.
    frames, gt = load_frames(), load_gt(GT)
    out = process_frames(frames, gt, alpha=24.0)
    labels = np.concatenate([lf.label for lf in out])
    glass = (labels == M.GLASS).sum()
    interior = (labels == INTERIOR).sum()
    assert glass > interior          # recessed pane is glass, not interior
    assert interior > 0              # but a real see-through tail remains
    assert (labels == M.NOT_GLASS).sum() > 0   # frames (mullions) now captured too


def test_one_result_per_frame_with_consistent_shapes():
    frames, gt = load_frames(), load_gt(GT)
    out = process_frames(frames, gt, alpha=24.0)
    assert len(out) == 3
    for lf in out:
        n = lf.xyz.shape[0]
        assert lf.xyz.shape == (n, 3)
        assert lf.normal.shape == (n, 3)
        assert lf.intensity.shape == (n,)
        assert lf.label.shape == (n,)


def test_dropped_points_removed_and_classes_present():
    frames, gt = load_frames(), load_gt(GT)
    out = process_frames(frames, gt, alpha=24.0)
    labels = np.concatenate([lf.label for lf in out])
    assert (labels == -1).sum() == 0                       # DROP cropped out
    assert (labels == M.GLASS).sum() > 0
    assert (labels == INTERIOR).sum() > 0
    kept = sum(lf.xyz.shape[0] for lf in out)
    raw = sum(f.xyz.shape[0] for f in frames)
    assert kept < raw                                      # cropping happened


def test_stamp_preserved_per_frame():
    frames, gt = load_frames(), load_gt(GT)
    out = process_frames(frames, gt, alpha=24.0)
    assert [lf.stamp_ns for lf in out] == [f.stamp_ns for f in frames]
