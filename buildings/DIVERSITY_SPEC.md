# Building Generation — Diversity Parameter Spec

Goal: generate a *rich, realistic* set of facades (GCC + Lebanon) for LiDAR segmentation
training by sampling from **one principled parameter space**, not by patching per building.

All lengths are **metres**. The generator emits labeled axis/face-oriented boxes; the spawner
divides scale by the calibrated `Cube` asset size (`spawn_calib.json`). Glass sub-type is kept
per point because the forward-model post-processing needs it (different a,b,m per glass).

---

## 0. How one building is sampled (the hierarchy)

```
region            (gcc | lebanon)                         -> weights everything below
 └─ building geometry   (footprint, floors, height, setbacks)   [§3]  -- orthogonal to facade
 └─ facade SYSTEM TYPE  (S1..S7, region-weighted)               [§1]  -- the core choice
      └─ facade composition params (drawn from the system's ranges)  [§2]
 └─ materials           (glass type, spandrel, frame finish)         [§4]
 └─ (capture sampling is orthogonal, applied at scan time)           [§6]
```

Per face, a system may still vary (a tower can be S1 on the shaft + S6/stone podium = §3 setback).

---

## 1. Facade system types (the core axis)

Sample a system per building (or per face for podiums), weighted by region (§5). Each system
fixes the *character*; §2 draws the exact numbers from the system's ranges.

| # | System | Character | Frame visibility (LiDAR) | Status |
|---|--------|-----------|--------------------------|--------|
| **S1** | Unitized curtain wall — **captured** | full-glass faces, pressure-plate mullion/transom grid proud 10–30 mm | strong grid | ✅ core (recessed-glass model) |
| **S2** | Unitized curtain wall — **SSG / flush** | full glass, ~15–25 mm silicone seams, ~flush | weak/thin seams | ⬜ todo (axis-aligned) |
| **S3** | **Stick-built** curtain wall | mid-rise, deeper/expressed mullions 50–100 mm | strong grid | ⬜ todo (axis-aligned) |
| **S4** | **Spider / point-fixed** | large clear panels on discrete spider fittings, no continuous frame | sparse points (fittings) | ⬜ todo (needs rotated geom) |
| **S5** | **Ribbon / strip windows** | horizontal vision bands between spandrel/wall bands | horizontal transoms strong, few mullions | ✅ `window_bands` style |
| **S6** | **Punched windows** in solid wall | discrete windows in concrete/stone/render wall (residential/Lebanon) | frame per window, wall dominant | ⬜ todo (axis-aligned) |
| **S7** | **Podium / base** override | different system on lower N floors (stone, stick, storefront) | varies | ⬜ todo (uses §3 setback) |

### Per-system parameter ranges

| param | S1 captured | S2 SSG/flush | S3 stick | S5 ribbon | S6 punched |
|-------|-------------|--------------|----------|-----------|------------|
| horizontal module width (m) | 1.2–1.8 | 1.2–1.8 | 0.9–1.5 | 1.2–2.0 | bay 2.5–5.0 |
| mullion face width (m) | 0.050–0.065 | 0.015–0.025 (seam) | 0.060–0.100 | 0.050–0.080 | 0.05–0.09 (window frame) |
| frame depth (m) | 0.04–0.06 | 0.02–0.03 | 0.05–0.12 | 0.04–0.06 | 0.05–0.10 |
| frame offset vs glass | proud/recessed | ~flush | proud | recessed | flush in reveal |
| `_RECESS` glass setback (m) | 0.04–0.08 | 0.00–0.02 | 0.05–0.10 | 0.04–0.08 | 0.05–0.15 (deep reveal) |
| vision fraction / floor | 0.55–0.85 | 0.60–0.90 | 0.50–0.70 | 0.30–0.55 (band) | n/a (window h×w) |
| intermediate transom | ~60% | ~40% | ~70% | n/a | per-window head/sill |
| glazed-column fraction | 0.85–1.0 | 0.9–1.0 | 0.7–0.95 | 1.0 (banded) | window-to-wall 0.2–0.4 |
| wall / solid fraction | low | low | low–med | low | **dominant** |

---

## 2. Facade composition parameters (per face, within a system)

| param | range / note | status |
|-------|--------------|--------|
| floor-to-floor height | office 3.5–4.5 m; residential 2.8–3.2 m | ✅ (office) |
| n columns / bays per face | `round(face_width / module_width)` | ✅ |
| window **width** | bay − mullion (curtain) **or** explicit 1.0–1.8 m (punched) | ⚠ implicit; add explicit |
| window **height** | vision_fraction·floor **or** explicit 1.2–2.6 m | ⚠ implicit; add explicit |
| sill height (spandrel below) | (1−vision)·floor, or 0.6–1.1 m punched | ✅ |
| spandrel band height | floor − vision (+ slab zone) | ✅ |
| floor-line transom | always | ✅ |
| intermediate transom (vision/spandrel) | per §1 toggle | ✅ (toggle) |
| mixed face: glazed vs wall columns | `wall_col_prob` 0.0–0.35; `full_wall_face_prob` 0.05 | ✅ basic; enrich |
| contiguous wall **regions** (wall between 2 grids, half-half) | 10–40% of faces | ⬜ todo |
| corner treatment | corner post (square) / mitred (angled) | ✅ post; ⬜ mitred |
| reveal / shadow depth | from `_RECESS` | ✅ |

---

## 3. Building geometry (orthogonal to facade system)

| param | range | status |
|-------|-------|--------|
| **footprint shape** | rectangle, square, L, U, T, notched, chamfered-corner, angled-polygon | ✅ rect/square + chamfer (geom); ⬜ **spawn rotated** |
| footprint size | 12–40 m per side (GCC towers larger; Lebanon 10–25 m) | ✅ |
| # floors | 5–12 (extend to 30+ for GCC towers) | ✅ 5–10 |
| floor height | per §2 | ✅ |
| **setbacks / podium+tower** | podium 1–3 floors larger footprint, tower above | ⬜ todo (enables S7) |
| parapet / roof edge | 0.5–1.5 m upstand; flat roof slab | ⬜ todo |
| ground slab + standoff context | present | ✅ |

**Key prerequisite:** angled footprints + L/U faces + spider = **rotated boxes**. The current
spawner only validated axis-aligned (square) boxes. Must nail UE5's object-rotation convention
(spawn a box at a known R, measure its true orientation, lock it) before any non-rectangular
geometry. This is the single gate for §3 shapes and S4.

---

## 4. Materials / glass (per building; feeds the labels)

| param | options | note |
|-------|---------|------|
| vision glass type | clear/tinted (R≈4–8% @905nm), solar-control coated (20–50%), low-E (50–80%) | one type/building (spandrel may differ); region-weighted |
| glass tint | clear, bronze, grey, green, blue | affects transmittance, not 905 nm reflectance much |
| spandrel type | ceramic-frit, shadow-box, opacified | all → `spandrel` label |
| frame finish | anodised / powder-coat aluminium | all → `metal_frame` label |
| wall cladding | concrete, stone, render, metal panel | → `wall` label |

Labels (8 classes, `forward_model/materials.py`): wall, metal_frame, glass_clear,
glass_coated, glass_lowE, spandrel, ground, other.

---

## 5. Region priors

| | GCC | Lebanon |
|---|-----|---------|
| system mix | S1 0.45, S2 0.30, S3 0.10, S5 0.10, S6 0.05 | S6 0.35, S3 0.20, S5 0.20, S1 0.15, S2 0.10 |
| vision glass | coated 0.55 / low-E 0.35 / clear 0.10 | clear 0.50 / coated 0.35 / low-E 0.15 |
| height | taller (towers) | mid-rise |
| footprint | mostly rectangular towers | more varied / angled / smaller |
| wall cladding | rare (mostly glazed) | common (stone/render) |

(System-mix numbers are tunable engineering priors, not measured statistics.)

---

## 6. Capture sampling (orthogonal — applied at scan time, already built)

Per face: 1 m position grid (1.5 m if face > 600 m²); **K=4** shots/cell, each with
standoff biased **1–2 m** (range 1–4), pitch biased **0…−20°** (range ±40), roll/yaw ~0
(range ±40/±30). Raw save per shot: `xyz, label, normal, sensor_pose`. (`buildings/orbit.py`,
`capture.py`.)

---

## 7. Implementation status & build order

**Done (solid, axis-aligned):** rectangular/square AND **L/U/T** multi-face buildings at correct
scale (winding-order outward normals for concave shapes); all 5 systems S1/S2/S3/S5/S6; mullion/
transom grid + intermediate transom; spandrel; flush-glass + proud-frame depth model; **window-size
control** (`window_width_frac` inset + wider bays) ; **protruding horizontals** (string-course
bands + parapet + plinth, `_add_horizontals`); mixed glazed/wall columns; corner posts; per-point
exact normals; region glass mix; orbit capture + raw save (verified offline); RGB viz.

**Next (axis-aligned → robust, no convention risk):**
1. ~~Window style S2/S3/S6~~ ✅ ; ~~explicit window width×height~~ ✅ ; ~~parapet/bands~~ ✅.
2. Frame-style variants via depth knobs (flush / protruding / expressed) — partial (per-system).
3. Mixed-face **wall regions** (wall-between-grids, half-half) — de-prioritised by user.
4. **Capture budgeting**: spread poses across faces + per-building cap (currently front-loads face 0).

**Gated on rotated-box spawn fix:**
5. UE5 object-rotation convention (measure + lock) — the prerequisite.
6. Angled/chamfered footprints, mitred corners, S4 spider, S7 podium setbacks.

**Then:** scene clutter/distractors (generalization) → generate the real multi-building pilot batch
→ apply the calibrated forward model (`CALIBRATION.md`) → PointNet++.

Each numbered item = a small, testable change to `buildings/build.py` driven from this spec's
ranges; the spec is the single source of truth for "what varies and by how much."
