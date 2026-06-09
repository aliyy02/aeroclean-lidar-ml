# Results — PointNet++ semantic segmentation (first run)

First real run of the segmentation model on the labelled Unitree-L2 scans. The model is
**PointNet++ (`pointnet2_sem_seg`, TF1.x)** trained for **4 classes** on **8192 points/scene**
with **4 input features (x, y, z, intensity)**. Cross-validation, **fold 0**; best validation
checkpoint at **epoch 065**.

Label scheme (same as the rest of the repo): `0 = not_glass (frame/mullions)`, `1 = glass`,
`2 = ground`, `3 = interior`.

## Held-out test metrics (best model)

| Metric | Value |
|--------|------:|
| Point accuracy | **0.873** |
| Mean class accuracy | 0.840 |
| Mean precision / recall / F1 | 0.847 / 0.840 / 0.842 |
| **Mean IoU** | **0.740** |

Per-class:

| Class | Precision | Recall | F1 | IoU |
|-------|----------:|-------:|----:|----:|
| glass | 0.882 | 0.974 | 0.926 | **0.862** |
| ground | 0.937 | 0.951 | 0.944 | 0.894 |
| interior | 0.832 | 0.767 | 0.798 | 0.664 |
| not_glass (frame) | 0.739 | 0.666 | 0.701 | 0.539 |

Confusion matrix (rows = true, cols = predicted):

| true \ pred | not_glass | glass | ground | interior |
|-------------|----------:|------:|-------:|---------:|
| **not_glass** | 63873 | 13384 | 5091 | 13531 |
| **glass** | 2954 | 138274 | 67 | 632 |
| **ground** | 5879 | 1910 | 212868 | 3158 |
| **interior** | 13747 | 3125 | 9206 | 85741 |

**Read of the numbers:** glass is segmented cleanly (97.4 % recall, IoU 0.86) and ground is
easy (IoU 0.89). The two hard classes are **frame** (IoU 0.54) and **interior** (0.66): the
frame is thin and gets confused with both glass and interior, and interior (deep see-through
returns) blurs into frame/ground. That matches the physics — frame and interior are the sparse,
geometrically ambiguous classes.

(Best-epoch validation, for reference: mean IoU 0.763 — glass 0.851, ground 0.924, interior
0.708, frame 0.569.)

## Layout

```
results/
├── training/
│   ├── log_train.txt   full training log (per-epoch loss/acc + the final test evaluation above)
│   ├── train.py        training script (friend's TF1.x PointNet++ fork — see note below)
│   └── model.py        model wrapper (imports models/pointnet2_sem_seg)
└── predictions/
    ├── oxy/            prediction on a held-out Oxy (coated-diffuse) scan
    │   ├── oxy_test_prediction_3d.png             static 3-D render of the predicted labels
    │   ├── oxy_test_prediction_3d_interactive.html  ← open in a browser to rotate it
    │   ├── oxy_test_group_random5.npz             model input  (xyz, intensity)  [20480 pts]
    │   └── oxy_test_prediction_random5.npz        model output (points, pred_label, confidence, probabilities)
    └── bech/           prediction on 5 held-out Bech (tinted / see-through) frames
        ├── bech_5_prediction_3d_interactive.html  ← open in a browser to rotate it
        ├── frames/     the 5 input frames WITH ground-truth labels (xyz, intensity, label, normal)
        ├── index.csv   per-frame point + class counts
        └── meta.json   plane-correction params used when the frames were labelled
```

## The two held-out buildings

These are two *new* buildings the model had not seen, run for a qualitative look:

- **Oxy** (coated-diffuse glass): only the prediction is shipped (no ground truth). Predicted
  mix ≈ glass 5.9k / interior 10.0k / frame 4.4k / ground 0.2k of 20.5k points, mean confidence
  0.62.
- **Bech** (tinted, see-through glass): ground-truth labels are included, so this one is
  checkable. The predicted class mix tracks the ground truth closely on the dominant classes
  (ground 10385 vs 10077 GT, frame 5201 vs 4639 GT); glass is under-predicted (48 vs 182 GT),
  which is expected — Bech glass is mostly saturated/sparse and only ~1 % of points.

> One of these two is the clean case and one is the harder/failure case — see the interactive
> HTMLs. Final "good vs failure" tagging is pending the author's call.

## What's **not** in the repo (and where it lives)

To keep the repo light, these were intentionally left out:

- **Trained weights** — ~140 MB of TF checkpoints (`*.ckpt.data/.index/.meta`, multiple epochs).
  Live in the training run on Colab Drive: `pointnet2_runs/run_01/fold0/`.
- **TensorBoard event files** (`events.out.tfevents.*`) — the loss/accuracy curves.
- **Raw bag** — the 71 MB `.mcap` for the Oxy test scan.

## Note on running `train.py`

`train.py` / `model.py` are kept as a record of *how* the model was trained (hyper-params,
4 classes, 8192 pts, Adam @ 1e-3). They are **not standalone-runnable from this repo**: they
import `models/pointnet2_sem_seg`, `provider`, `tf_util`, and `data_prep/scannet_dataset` from
the friend's PointNet++ fork, which is not vendored here.
