# calibration/ — real-scan labeling

Turn a **real Unitree-L2 bag** + its **`test_bed_multiwindow.py` ground-truth `.txt`** into
**per-frame labeled point clouds** for fitting the forward model (and a gold test set).

This is the read→align→segment stage of the calibration plan in `../CALIBRATION.md`; the fit
stage (θ → constants) comes after the roll is locked across all bags.

## Run

```bash
cd ml_pipeline
source /opt/ros/jazzy/setup.bash          # needed only to read the bag
python3 -m calibration.label_scan  <bag_dir>  <gt.txt>  <out_dir>  --alpha 24 --qa
# e.g. python3 -m calibration.label_scan L6_test_1 L6_test_1.txt data/labeled/L6_test_1 --qa
```

Outputs in `<out_dir>`: `frame_NNNN.npz` per frame, `meta.json`, `index.csv`, and (with `--qa`)
`qa_overlay.png`.

```python
import numpy as np
d = np.load("data/labeled/L6_test_1/frame_0000.npz")
xyz, intensity, label, normal = d["xyz"], d["intensity"], d["label"], d["normal"]
# label ids: 0 not_glass, 1 glass, 2 ground, 3 interior (see-through)
```

## Label all bags + view in RViz

The 43 L6 scans live in `L6_all/L6_all/L6_test_N/` with GT in `L6_tests_1/L6_tests_1/L6_test_N.txt`.
Label them all (one α=24 for every bag — the mount roll is a constant sensor property):

```python
from calibration.label_scan import label_scan
for n in range(1, 44):
    label_scan(f"L6_all/L6_all/L6_test_{n}", f"L6_tests_1/L6_tests_1/L6_test_{n}.txt",
               f"data/labeled/L6_test_{n}", alpha=24.0, qa=True)
```

Inspect the labels in **RViz** (colored cloud + GT window outlines, gallery side-by-side):

```bash
source /opt/ros/jazzy/setup.bash
python3 -m calibration.rviz_publish --bags 1,7,20,25,43        # any subset
rviz2 -d calibration/labeled.rviz                              # fixed frame: map
```

Colors: glass=cyan, interior=orange, not_glass=gray, ground=brown; red boxes = GT windows.
The view frame is z-up (`X right, Y forward, Z up`). Each bag's `data/labeled/<bag>/qa_overlay.png`
is a quick 2D check without RViz.

## What it does

```
native L2  ──(ned = sz, sx, sy)──▶  body-NED  ──R_x(α)──▶  corrected NED (== GT frame)
   then, per point:  intersect the beam with the facade plane (arbitrary tilt — the scans
                     sweep pose, so the facade is generally NOT at constant x); the real panel
                     sits ~16 cm BEHIND the measured plane (panel-depth offset, auto-estimated)
                     glass     : in a window aperture, in the panel depth zone (the recessed pane)
                     interior  : in a window aperture, deeper than interior_cut (deep see-through)
                     not_glass : in the panel zone, in the grid, not a window (mullion/frame)
                     ground    : on the floor plane, in the panel's lateral extent
                     (dropped) : everything beyond the window grid in u/v
```

Label ids 0/1/2 reuse `forward_model.materials` so real data shares the sim's label space;
`interior=3` is real-scan-only (the deep see-through tail). Normals are geometry-exact
(facade `[-1,0,0]`, ground `[0,0,-1]`).

## Three real-vs-`test_bed` facts (found on the L6 scans)

1. **Axis map** native→body-NED is `ned = (sz, sx, sy)` (forward = native z) — brute-force-confirmed;
   differs from the sim's `_des_to_ned`.
2. **Boresight roll ≈ 24°** between the L2's true axes and the mounting-hole frame `test_bed.py`
   assumes — **not modeled there** (config knob `alpha`).
3. **Panel-depth offset ≈ 16 cm**: the real panel sits ~16 cm behind the plane the measured corners
   define (accumulated ruler/tape error in the facade geometry, the same for all bags). So the bulk
   of in-window returns is the **recessed glass pane** (labeled glass), not see-through. Auto-measured
   per bag by `estimate_panel_offset()` → `interior_cut`; the deep tail past it is interior. Typical
   label mix: glass ~62%, ground ~26%, frame ~8%, interior ~4%.

## Tuning the roll `α`

`--alpha` (default 24) is a **config knob**. The auto `estimate_roll()` helper is only a rough
initializer — on a glass-filled panel embedded in a larger coplanar wall the grid boundary is masked,
so **set α from the `--qa` overlay**: it back-projects the facade to the glass plane and overlays the
GT windows; pick the α where they line up. A constant sensor-mount roll should give the **same α on
every bag** — a good cross-check once more bags are in.

## Modules / tests

`gt_parse` (GT `.txt` → windows + plane), `frames` (remap + roll), `roll_estimate` (rough α),
`label` (the per-point rules + floor detect), `io_bag` (mcap reader, ROS), `label_scan` (CLI + QA).
Tests are offline (`python3 -m pytest calibration/ -q`); a 3-frame slice of the real bag is cached
under `tests/fixtures/` so the end-to-end test needs neither ROS nor the sim.

Design spec: `../docs/superpowers/specs/2026-06-07-realscan-labeling-design.md`.
