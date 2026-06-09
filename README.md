# aeroclean-lidar-ml

**Learning to see glass — LiDAR semantic segmentation for autonomous facade cleaning.**

Part of the **AeroClean** project: a drone that cleans high-rise glass facades. To clean a
window it must locate each pane to within a few centimetres relative to the building. Glass is
nearly invisible to most sensors, so this repo builds the perception that makes it possible.

The idea: instead of a hand-tuned geometric pipeline, **classify every LiDAR point** into
`glass / frame / interior / ground`, then read the window corners off the glass points. The
classifier (PointNet++) is trained on real scans plus **physically simulated** ones — a
first-principles forward model paints realistic glass returns onto clean simulator geometry.

```
  buildings/        procedural building → spawn in CosysAirSim/UE5 → scan → raw geometry + labels
  forward_model/    per-beam physics: reflectance, return probability, saturation, see-through
  calibration/      label REAL Unitree-L2 scans against ground-truth window corners
  analysis/         exploratory data analysis (EDA) + the figure generator
  test_bed.py       the 6-DOF rig kinematics (real ground-truth capture)
```

## What's here

| Path | What it does |
|------|--------------|
| `buildings/` | Procedural multi-face buildings (several facade systems; square / L / U / T footprints) → spawn → orbit capture → raw `.npz` (geometry + labels, no intensity). |
| `forward_model/` | First-principles LiDAR optics: reflectance ρ(θ), return-probability P(θ), 255 saturation, transmission → interior returns. Turns clean sim geometry into realistic intensity + sparse glass. |
| `calibration/` | Labels real L2 scans: fit the true glass plane, project points perpendicularly, region-grow → `glass / frame / interior / ground`. |
| `analysis/` | EDA tools and `make_slides.py` (regenerates every figure in `presentation/figures/`). |
| `presentation/figures/` | EDA / glass-characterization figures (catalogued in `figures/INDEX.md`); regenerate with `analysis/make_slides.py`. |
| `docs/` | `calibration.md`, `glass_characterization.md` (the glass EDA), reference PDFs. |
| `examples/` | A few tiny labelled sample frames + a ground-truth file, for a quick look. |
| `results/` | First PointNet++ run (fold 0): training log + model scripts + predictions on two held-out buildings (interactive 3-D). Weights/bags excluded — see `results/README.md`. |

## Quick start (offline, no simulator)

```bash
pip install -r requirements.txt

python3 -m pytest -q                      # 158 tests (buildings + forward_model + calibration)

python3 -m buildings.render_gallery       # preview procedural buildings → /tmp/building_gallery.png
python3 analysis/make_slides.py           # (re)build the EDA figures → presentation/figures/
```

Peek at a labelled sample frame:

```python
import numpy as np
d = np.load("examples/L6_sample_frame.npz")
# keys: xyz (N,3), intensity (N,), label (N,)  0=frame 1=glass 2=ground 3=interior,  normal (N,3)
```

## Live (with CosysAirSim + UE5 in Play, ComputerVision mode)

```bash
python3 -m buildings.calibrate_spawn                         # once: pin the cube asset size
python3 -m buildings.ros_publish --square-prob 0.6 --region gcc   # drive a facade in RViz
python3 -m buildings.capture --episodes 1 --max-caps 5 --out data/raw_test
```

## Data

The full datasets are **not** in this repo (1.7 GB labelled clouds + ~1.6 GB raw `.mcap`
bags). Three real buildings were scanned with a 6-DOF rig — **L6** (43 bags), **Oxy** (59),
**Bechtel** (40) — 142 scans total, labelled against tape-measured window corners. See
`docs/calibration.md` for the labelling method and `docs/glass_characterization.md` for what
the three glass types actually do to a LiDAR. A handful of sample frames live in `examples/`.

## Results

First PointNet++ semantic-segmentation run on the labelled L2 scans (cross-val fold 0):
**test mIoU 0.74**, point accuracy **0.87** — glass IoU **0.86**, ground 0.89 (frame 0.54 and
interior 0.66 are the confusable classes; glass recall 97 %). The full training log, the model
scripts, and predictions on two held-out buildings (Oxy, Bech — interactive 3-D HTML you can
rotate) are under `results/`; see `results/README.md`. Trained weights (~140 MB of TF
checkpoints) and raw bags are intentionally excluded.

## Figures

The EDA / glass-characterization figures live in `presentation/figures/` (catalogued in
`presentation/figures/INDEX.md`). Regenerate them all with `python3 analysis/make_slides.py`.

## Status

Built and tested: building generator, forward model, real-scan labelling (158 unit tests).
**Pending:** PointNet++ inference into this repo + corner extraction from the segmentation, and
cross-facade validation of the forward model.

— AeroClean · EECE 490 (Applied Machine Learning), AUB
