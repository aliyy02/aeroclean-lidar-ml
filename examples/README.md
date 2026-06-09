# Example data

Tiny samples so you can poke at the data without the full datasets.

- `L6_sample_frame.npz`, `Oxy_sample_frame.npz`, `Bech_sample_frame.npz` — one labelled LiDAR
  frame from each building.
- `L6_ground_truth.txt` — a ground-truth file (rig pose + per-window corner geometry) as
  produced by `test_bed_multiwindow.py` and parsed by `calibration/gt_parse.py`.

Each `.npz` has:

| key | shape | meaning |
|-----|-------|---------|
| `xyz` | (N, 3) | point position in corrected NED, metres (x = depth from sensor) |
| `intensity` | (N,) | return strength, 0–255 |
| `label` | (N,) | `0` frame · `1` glass · `2` ground · `3` interior |
| `normal` | (N, 3) | surface normal at the point |

```python
import numpy as np
d = np.load("examples/L6_sample_frame.npz")
print(d["xyz"].shape, np.unique(d["label"]))
```
