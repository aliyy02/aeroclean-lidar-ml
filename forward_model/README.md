# forward_model — first-principles LiDAR intensity & return synthesis

The simulator (Standard Lidar, SensorType 6, ComputerVision mode) returns **geometry
only**: per-beam XYZ + the UE5 actor name it hit. This package adds the sensor's optics
**per point, in pure Python**, so synthetic scans look like a real Unitree L2 scan:
distance falloff, incidence-angle intensity, a glass return cone, a noise-floor cull,
and position jitter. The constants are calibrated once against real test-bed scans; the
code never changes (calibration only swaps numbers).

## The model (per point)

```
R       = ||point||                                   # range
theta   = angle(beam, surface_normal)                 # incidence (normal from the manifest)
rho_eff = a*cos(theta) + b*exp(-tan^2(theta)/m^2)/cos^5(theta)   # Layer 3 (== rho_eff)
P_r     = C * rho_eff / R^n                            # Layer 1 (C/R^n ONLY -- no 2nd cos)
keep    = P_r > T                                      # Layer 4 threshold
          and (theta <= cone_half_width if glass)      # Layer 2 monostatic cone
xyz    += Gaussian(sigma)                              # Layer 4 jitter
intensity = P_r                                        # reported count ∝ received power
```

`a`/`b` are **independent** (diffuse ~ rho/pi; specular ~ Fresnel·G). `cos^5` (not cos^4)
is the assembled monostatic intensity. See `../lidar_post_processing.pdf` for the derivation.

## Usage

```python
import numpy as np
from forward_model import apply_forward_model, load_constants

constants = load_constants("forward_model/constants_calibrated.yaml")  # or None for priors
res = apply_forward_model(xyz, names, normals, constants, rng=np.random.default_rng(0))
# res.xyz (K,3)   surviving points (jittered)
# res.intensity   received power per point (∝ L2 intensity count)
# res.labels      class id per point (see materials.CLASS_NAMES)
```

`normals` is an (N,3) per-point surface normal resolved from the scene manifest by the
caller (the scene generator records each actor's plane normal).

## Files

| File | Role |
|------|------|
| `materials.py` | actor-name prefix -> class id (8 classes) |
| `geometry.py` | range + incidence angle |
| `intensity.py` | Layer 3 `I(theta)` (== rho_eff) |
| `range_model.py` | Layer 1 `C*rho_eff/R^n` (single-cos rule) |
| `returns.py` | Layer 2 cone gate, Layer 4 threshold + jitter |
| `ghosts.py` | optional Householder mirror points |
| `constants.py` | named placeholder priors + calibrated-YAML loader |
| `constants_calibrated.yaml.template` | the contract your calibration fills in |
| `apply.py` | orchestrator: scan -> (xyz, intensity, labels) |

## Calibration handoff

Calibration is owned separately (the 6-DOF `test_bed.py` rig). It fits `a,b,m,n,
cone_half_width` per material + `C,T,sigma` and writes `constants_calibrated.yaml`
(schema = the template). Until then, the physically-ordered priors in `constants.py`
produce sane (if not yet L2-matched) synthetic data.

## Tests

```
python3 -m pytest forward_model/ -q     # 44 offline unit tests, no sim needed
```
