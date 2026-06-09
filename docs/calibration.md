# Forward-model calibration — protocol for fitting on real LiDAR scans

**Goal:** turn the real Unitree-L2 test-bed scans into numbers in
`forward_model/constants_calibrated.yaml`, so the synthetic intensity / return-probability
matches the real sensor. **The forward-model CODE does not change — only these constants do.**

This file is the brief for whoever (a fresh Claude session) does the fit. It states the inputs,
the labeling convention, exactly what is fittable, the procedure, and the output. The tooling
(`calibration/`) is **not built yet** — build it against the real data format once it's in hand.

---

## 0. Inputs (what Ali provides)

- **~43 scans**, each: the LiDAR point cloud **in the LiDAR frame** + the **full ground-truth
  geometry in a reference frame** (window/pane rectangles, frame thickness, protrusion, and a
  best-guess material for frame and glass). A **single rigid transform** (R, t) aligns the two
  frames (from the 6-DOF rig kinematics, or solved from known corner correspondences — verify it
  by checking GT window corners land on the cloud, report the residual).
- **Per-point labels** applied by Ali using the convention below.
- Scene: a real multi-window panel; the rig sweeps pose, standoff, and **incidence angle**.

## 1. Labeling convention (decided 2026-06-07 — keep sim + real identical)

**Label by the return MECHANISM, not the nominal front surface:**
- **glass** = a return off transparent glazing with **air/interior behind it** (sparse, faint).
- **frame / not-glass** = any solid diffuse return: the metal cap **and the pane-edge zone where
  metal sits directly behind the glass** (the beam goes through the glass, reflects off the metal,
  returns ≈20% dimmer at the metal's range — physically a metal return, so labeled frame).
- Use the GT geometry to draw the line "glass ends where air-behind ends." Expect a thin
  ambiguous seam (one beam straddling) → assign by majority.

**Consistency hook for the synthetic side:** the real "frame" stripe is *wider* than the visible
metal cap (it includes the behind-glass edge). Measure that full frame-label width from the GT and
set the generator's `mullion_face` to match, so the 2D frame/glass coverage agrees sim↔real. Lean
on **intensity** (metal-bright vs glass-faint), not relative depth, as the cross-domain cue.

## 2. What is fittable vs. prior-only

The model per point (`forward_model/`): `I(θ) = a·cosθ + b·exp(-tan²θ/m²)/cos⁵θ` (== ρ_eff),
`P_r = C·ρ_eff / Rⁿ`, glass returns only inside a near-normal cone, keep iff `P_r > T`, add
jitter σ. θ is the **incidence angle** = angle(beam, KNOWN surface normal from GT) — exact, no PCA.

| Constant | Fit from | Notes |
|----------|----------|-------|
| `a`, `b` per material | intensity vs θ (per labeled material) | only the products `C·a`, `C·b` are observable; fix overall scale to L2 counts. `b` must rank lowE > coated > clear |
| `m` (glass lobe width) | intensity vs θ near θ=0 | **needs dense sampling ≤1–2° near normal** — the peak is ~2–5° wide; coarse sweeps alias past it |
| `n` (range exponent) | the x/y/z **range sweep** | expect ~2 for glass/wall; steeper for thin frame bars |
| `cone_half_width` (glass) | **return-RATE** vs θ for glass | the angle past which glass stops returning |
| `T` (threshold) | per-material **return-rate** per θ-bin | rosbags hold only returns → fit T as a RATE, not absolute power. Reconstruct the L2 nominal beam grid (spec or union of all returns) as the denominator |
| `σ` (jitter) | residual of flat **frame/wall** points to their plane | **not glass** — glass spread includes real multi-pane (IGU) structure, not sensor noise |

**Prior-only (don't try to fit from the bed):** absolute `C` (folded into a,b); glass classes you
didn't physically scan (keep priors; cross-check by fitting `b` and seeing which coating class it
implies); Beer–Lambert α and ghost conditions (low confidence — defer).

**Numerical guards:** `cos⁵θ` blows up near θ→90° — clip / work in log space; the exp() factor
drives it to 0 anyway.

## 3. Procedure (build `calibration/` to do this)

1. `read_scan` — load each scan's cloud (LiDAR frame) + its GT geometry + labels.
2. `align` — apply/solve the rigid (R, t) to put cloud and GT in one frame; assert window corners
   land inside the GT quads; **report the residual** before trusting any θ.
3. `segment` — from GT + labels, tag each point glass / frame / (background) per §1.
4. `theta` — per point, θ = angle(beam direction, GT surface normal). Exact.
5. `fit` — per material: normalize intensity by `Rⁿ`; nonlinear least-squares for `(A=C·a, B=C·b, m)`
   over the θ sweep; fit `n` from the range sweep; glass `cone_half_width` + `T` from return-rate
   vs θ; `σ` from flat frame/wall plane residuals.
6. Write `forward_model/constants_calibrated.yaml` (schema = `…yaml.template`). **Report a
   confidence per constant and flag any that fell back to priors.**

## 4. Output + verification

- Output: `forward_model/constants_calibrated.yaml`. `forward_model.load_constants(path)` picks it
  up; any omitted key keeps its prior. Nothing else changes.
- Sanity: `b` larger for coated/low-E than clear; glass cone ~2–5°; `n`≈2 for glass/wall.
- End-to-end: run `forward_model.apply()` over a **synthetic** building capture with the new
  constants and confirm glass goes sparse with a near-normal bright patch, frame/wall/ground dense
  and diffuse — i.e. it now looks like the real scans.

## 5. Pointers

- Physics + API: `forward_model/README.md`, `forward_model/apply.py`, `…/intensity.py`,
  `…/range_model.py`, `…/returns.py`, `…/geometry.py`.
- Output contract: `forward_model/constants_calibrated.yaml.template`.
- The rig: `test_bed.py` (kinematics give the per-pose window normal + corners in the LiDAR frame).
- Model derivation + provenance: `lidar_post_processing.pdf`. Facade/optical priors: `Window Types.pdf`.
