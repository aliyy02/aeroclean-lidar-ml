# Glass characterization (L6 / Oxy / Bechtel)

Consolidated from the three working notes produced during calibration.
Section 1 is the cross-dataset comparison (the authoritative summary);
Appendices A–B preserve the L6-specific behavior and the model-diversification notes.

---

# Three glass types to the Unitree L2 — L6 vs Oxy vs Bech (evidence-based)

Data-derived comparison of **three** architectural glasses seen by the **Unitree L2 (~905 nm)**,
from the real test-bed scans. Built to **test, not assume**, the prior single-glass (L6)
conclusions. Every claim below is backed by numbers measured from the bags (tooling in
`analysis/`, figures in `analysis/*.png`). **Where this disagrees with `GLASS_BEHAVIOR_L6.md` /
`GLASS_MODEL_DIVERSIFICATION.md`, this file is the corrected version.**

Datasets: **L6** 43 scans, 2×4 win; **Oxy** 59 scans, 2×6 win (1.485×2.195 m), standoffs ~1.10 &
~1.51 m; **Bech** 40 scans, 2×3 win (1.46×1.882 m), standoffs ~1.29 & ~1.73 m. All swept over
yaw/pitch/roll (Bech to ±60° roll) → incidence-angle coverage 0–75°.

---

## TL;DR
- **Frame conventions HOLD for both new datasets.** Axis map `ned=(native_z, native_x, native_y)`
  (ysign +1) and boresight roll **α≈24°** are unchanged (same rig). Verified, not assumed: the
  fitted facade-plane normal matches the GT normal to **<1° median across all tilted poses**
  (L6 0.8°, Oxy 0.5°, Bech 0.3°). The mount did **not** change.
- **The "panel offset ≈ 16 cm constant" story is wrong as a universal.** The glass-behind-frame
  depth is **building-specific geometry**, not a fixed measurement error: L6 ≈ 5 cm (flush),
  Oxy ≈ **29 cm** (deep, tight surface), Bech ≈ flush **+ a see-through tail to 2.5 m**.
- **"Oxy ≈ L6" — RETRACTED (see §9, user note).** The Oxy windows had a **curtain behind the
  glass**; the in-window depth is **bimodal** with **~71 % of returns at ~0.30 m behind = the
  curtain**, not the glass. So the "flat ~140 diffuse" signature I attributed to Oxy *glass* is
  mostly the **matte curtain**. Oxy glass is actually **transmissive** (it lets the beam reach the
  curtain); its own surface is the minority front layer. Treat Oxy as "transmissive glass + matte
  backing," not a coated glass.
- **Bech is a genuinely new (and the most interesting) type:** an **absorptive/tinted "black"
  glass** that is **dim at normal (~65–100, absorbs 905 nm), strongly see-through (returns from
  the interior to 2.5 m+), and turns into a near-mirror at grazing (→255 saturation)**. This is
  textbook dielectric Fresnel + transmission — and it **breaks the single smooth `ρ(θ)` model**.
- **Reported intensity is range-compensated (n≈0) for ALL three** — confirmed across the two
  standoffs of each. The model's split (range-independent intensity + 1/R² only in detection) is
  correct. **Intensity also SATURATES at 255** (Bech frame 20%, Bech glass 10%) — currently
  unmodeled.

---

## 1. Frame conventions (verified per dataset, from data)
- **Axis map**: brute-forcing all 48 signed axis permutations against the GT plane picks
  `ned_x = native_z, ned_y = +native_x, ned_z = +native_y` unambiguously for both Oxy and Bech
  (depth = native z). Same as L6. (`analysis/explore_raw.py`.)
- **Boresight roll α≈24°**: min-area-rect estimate gives Oxy 20–28° (median ~24, 5 frontal scans);
  Bech is noisier (the hairline mullions + see-through confuse the boundary), but the **physics
  quantities are roll-invariant** (see §method) so α need only be roughly right for the labels.
- **Registration proof** (the real test): fit the facade plane to the data per scan and compare
  its normal to the GT normal. Median angle **0.3–0.8°** across *all* poses, both datasets → the
  axis-map + α=24 chain is correct even on ±30–60° tilted scans. (`analysis/physics_dash.py`
  panel 6.)

## 2. The three glasses — intensity vs incidence angle (the money figure)
`analysis/angular_compare.png`. Reported intensity ≈ ρ(θ) (range-compensated). Medians, fine bins:

| glass | normal (0–2°) | floor (20–40°) | grazing (65–73°) | shape |
|-------|---------------|----------------|------------------|-------|
| **L6** | **255** | ~80 | ~100 | sharp specular spike + flat floor + weak grazing rise |
| **Oxy** | ~163 | ~130 | ~165 | **flat/diffuse**, gentle U, mild grazing rise, **no spike** |
| **Bech** | ~116 | **~65 (min)** | **255** | dark floor + **strong Fresnel grazing spike to saturation** |

Fit of `ρ(θ)=a·((1-g)cosθ+g) + s·fresnel_grazing(θ) + b·exp(-tan²θ/m²)/cos⁵θ` (the current
`forward_model/reflectance.py`), reported intensity counts, C folded in:

| glass | a (floor) | g (flat) | s (grazing) | b (burst) | m (deg) | R² |
|-------|-----------|----------|-------------|-----------|---------|-----|
| L6 | 123 | 0.28 | 256 | **139** | 6.4 | **0.94** |
| Oxy | 142 | 0.59 | 355 | 22 | 4.5 | **0.96** |
| Bech | 100 | 1.0 | 400* | 78 | 0.8 | **0.58** |

`*` hit the bound. **The model form fits the coated glasses (L6, Oxy) well; it fits Bech poorly** —
its flat-dark-floor → sharp-grazing-spike-to-saturation cannot be a single smooth curve, and the
low/mid-angle in-window intensity is the **interior** (see-through), not the glass surface, so a
single `ρ(θ)` conflates two mechanisms.

## 3. Depth / see-through structure (`analysis/angular_fit.png` bottom row)
Glass return depth **behind the frame front**, by dataset:
- **L6**: median ~5 cm, tight (p10–p90 ~6–20 cm). Surface returns, negligible see-through.
- **Oxy**: median ~**29 cm**, fairly tight (p10–p90 ~8–39 cm). A **recessed but solid surface**
  (coating returns the beam); only a small deep tail. Apparent depth *decreases* at grazing
  (reveal occlusion).
- **Bech**: surface ~flush; **huge see-through tail — p90 = 2.55 m** at low/moderate angle (the
  beam transmits, returns from the room interior), **collapsing to the front surface at grazing**
  (Fresnel surface reflection takes over). Depth and intensity are mutually consistent: deep+dim
  at normal, shallow+bright at grazing.

→ **Glass-behind-frame depth is building geometry (reveal), not a sensor offset.** The "interior"
return is **real and large for Bech**, essentially absent for L6, small for Oxy.

## 4. Range-compensation & saturation
- **n≈0 confirmed for all** (near-normal glass intensity vs standoff): Oxy 146@1.44 m vs 141@1.84 m;
  Bech 74@1.53 m vs 79@1.97 m. Raw 1/R² would change ~1.6×; observed ~flat. Reported L2 intensity
  is a range-normalized reflectivity. (`analysis/agg.py`.)
- **Saturation at 255**: L6 glass 1.1% / frame 1.9%; Oxy 0.4% / 1.3%; **Bech glass 9.7% / frame
  20.0%**. Strong specular/grazing/metal returns clip. **Must be modeled** (clip ρ→255).

## 5. Frame (not-glass) contrast — cross-domain cue holds, magnitude varies
Frame brighter than glass everywhere, but: L6 +37, Oxy +25, **Bech +104** (frame ~202 saturating,
glass ~98). Bech frame brightens to saturation with angle (very reflective metal); L6 frame is
specular-at-normal. Useful, but the magnitude is per-building.

---

## 6. Forward-model verdict — validate or rebuild?

**Validated (keep):**
1. **Two-channel design** — range-independent reported intensity ρ(θ) + separate 1/R² return
   probability. Confirmed (n≈0) on all three glasses.
2. **Reflectance form** `a·((1-g)cos+g) + s·fresnel_grazing + b·burst` — good (R²≥0.94) for **coated**
   glass (L6, Oxy) and for the flat-floor + grazing-rise behaviour that is real for every material.
3. **Frame brighter than glass** as a feature.

**Wrong / missing (fix):**
1. **No intensity clipping.** Add ρ ← min(ρ, 255). Material (esp. metal/Bech-grazing) saturates.
2. **See-through / interior return is unmodeled.** The biggest gap. For transmissive glass (Bech;
   to a lesser extent clear/tinted) the beam passes and returns **from the interior at depths up to
   2.5 m+**, with its own intensity. The current `tau` only drops/passes the beam; it never places
   a return at a realistic interior depth. **PointNet++ sees this 3D structure**, so the generator
   must emit interior returns: `with prob τ(θ): return at sampled interior depth (0–~3 m behind),
   intensity from a room-clutter draw; else surface return`.
3. **A single smooth `ρ(θ)` per glass conflates surface reflection with transmission+interior.**
   Bech proves it. Restructure glass as **two mechanisms**: (a) **surface** reflectance
   `ρ_surf(θ)` = dark diffuse floor + **Fresnel grazing rise + near-normal burst** (this is where
   the sharp grazing spike lives), and (b) **transmission** `τ(θ)` → interior return. Fit
   `ρ_surf` to the *grazing/near-front* returns, not the see-through bulk.
4. **Re-anchor the per-type params to THREE measured types** (normalize to peak 255), replacing the
   single `glass_lowE` guess:
   - **`glass_coated_specular` (L6)**: strong near-normal burst (b≈0.55), moderate floor
     (a≈0.45, g≈0.3), grazing lift (s≈high), low transmission.
   - **`glass_coated_diffuse` (Oxy)**: flat bright floor (a≈0.55, g≈0.6), weak burst (b≈0.08),
     grazing lift, low–moderate transmission. *(new archetype; the old table had no flat-diffuse
     coated row.)*
   - **`glass_tinted_fresnel` (Bech)**: dark floor (a≈0.30), **high transmission τ≈0.6–0.8 →
     interior returns**, **strong grazing Fresnel to saturation** (s high, clips 255), weak burst.
     This is the long-missing **specular/absorptive branch** — now data-anchored, not predicted.
5. **Glass recess depth is a generator geometry parameter** (L6 5 cm, Oxy 29 cm) — randomize the
   pane setback behind the mullion plane per window; don't bake a fixed offset into the sensor model.

**Net:** the model's *structure* is largely right for coated glass and its Fresnel/two-channel
design is well-supported — but it is **not** right for transmissive glass until the **interior-return
mechanism, saturation, and per-type (3-anchor) reflectance** are added. So: **validate the skeleton,
rebuild the glass branch around surface-vs-transmission.**

---

## 7. The PHYSICS behind the intensity (literature-grounded)
A monostatic lidar measures **backscatter**: light returned along the incoming ray. The angular
reflectance of any surface decomposes into **diffuse + specular**, and the standard remote-sensing
model for lidar intensity vs incidence is the **Lambertian–Beckmann** combination (diffuse
Lambertian + a Beckmann specular lobe) — and crucially, **the pure Lambertian `cosθ` law is only
valid to ~20°; past that it fails** [Sensors 21:2960]. Our data breaking `cosθ` is therefore
*expected*, not anomalous. The three terms in `forward_model/reflectance.py` map to real physics:
- **Diffuse floor `a·((1-g)cosθ+g)`** — Lambertian scattering off micro-roughness/coating; the
  flatness `g` is the **Oren–Nayar** roughening that makes rough surfaces flatter than cosine.
- **Specular burst `b·exp(-tan²θ/m²)/cos⁵θ`** — the **Beckmann/Cook–Torrance** microfacet lobe; for
  a *monostatic* sensor it only returns near **normal incidence (retroreflection)**, width `m`=roughness.
- **Grazing lift `s·F(θ)`** — the **Fresnel** reflectance of a dielectric: ~4% at normal rising
  toward **100% at grazing** (Brewster ~56° for n≈1.5) [Fresnel eqns; Cornell]. Real for all glass.

**Per glass, the mechanism:**
- **L6 = low-E / solar coated.** A thin metal/metal-oxide coating is **NIR-reflective** (published
  ~20–80% at 905 nm) [AccuCoat; Adv. Funct. Mater. 2025; US patent 11,977,154]. Smooth coating →
  strong **near-normal specular retro spike (→255)** on a high diffuse floor (returns at all angles)
  + grazing Fresnel. LiDAR-*friendly*.
- **Oxy = coated but rougher/more diffuse.** Same NIR-reflective coating idea, but micro-rough →
  high, **flat** Oren–Nayar floor with a **weak** specular spike. Bright and angle-flat.
- **Bech = body-tinted / absorptive, uncoated-like.** The tint **absorbs 905 nm** (Beer–Lambert) →
  **dark** (low diffuse). At low/mid angle the surface barely reflects (~4% Fresnel) so the beam
  **transmits and returns from the interior**; toward grazing **Fresnel reflectance climbs to ~1**,
  so the front surface becomes a **mirror (→255 saturation)**. Pure dielectric physics, no coating.

## 8. Return vs no-return PROBABILITY — vs angle, range, material
The detection side is the **lidar range equation**: received power for a diffuse target
`P_r ≈ P_t · ρ(θ)·cosθ / R² · (system const)`, and a return is registered only when
**`P_r > noise floor T`** [SPIE LiDAR Range Equation; calibvision; LeddarTech]. (The diffuse
round-trip falls as ~1/R²; for the *link budget* the effective signal-vs-range can be as steep as
1/R⁴.) This predicts three dependencies — all visible in the data (`analysis/return_prob.py`,
relative measure: a fixed forward beam-cone holds a fixed number of fired beams, so glass
returns/frame ∝ P_return×const; bags record **only returns**, so absolute rate isn't recoverable):

- **vs MATERIAL** — higher reflectance → higher P_r → returns more reliably / from farther
  (max range ∝ √ρ). Strong **coated** glass (L6, Oxy) returns densely everywhere; **dark tinted**
  glass (Bech) returns weakly; clear glass mostly transmits.
- **vs INCIDENCE ANGLE** — two effects. (a) **Geometric**: a specular surface sends the beam *away*
  from a monostatic receiver except near **normal** (retro) or **grazing** (Fresnel back-scatter +
  roughness); a diffuse/coated surface returns at *all* angles. (b) **Radiometric**: `P_r∝ρ(θ)cosθ`.
  Measured: L6/Oxy return-density is **flat 0–40°** (Oxy flat even to 66°) → coated glass keeps a
  return at every angle; **Bech declines** with angle in the forward cone (dark surface, transmits).
- **vs RANGE / STANDOFF** — `P_r∝1/R²`, so it matters **only for weak returns near the floor**.
  Measured at near-normal across the two standoffs: **Oxy 738 (R≈1.4 m) = 738 (R≈1.8 m) — flat**
  (strong returns, far above T); **Bech 643 (R≈1.5 m) → 214 (R≈2.0 m) — ~3× drop** (weak see-through
  returns sink under T as 1/R² bites). **Key distinction:** reported *intensity* is range-compensated
  (n≈0, §4), but whether a beam *returns at all* rides on raw received power vs the noise floor,
  which **is** range-sensitive — two different quantities, and Bech shows the difference.

→ For the forward model the return channel is therefore: `P_return(θ)` (≈1 for coated; angle/Fresnel
shaped for tinted) **AND** a `C·ρ(θ)/R^n > T` noise-floor gate (the only place range lives) — both
already in `forward_model/detect.py`; the data confirms the structure and says **fit a low T for
coated glass (range-insensitive) and a higher effective T / lower C for dark glass (range-sensitive)**.

### References
- Lambertian–Beckmann intensity vs incidence; cosθ valid to ~20°: *Analysis and Radiometric
  Calibration for Backscatter Intensity … Incident Angle Effect*, Sensors 2021, 21:2960
  (doi.org/10.3390/s21092960); *Radiometric Calibration … Hyperspectral LiDAR Backscatter*,
  Remote Sens. 12(17):2855.
- Fresnel reflectance 4%→100% toward grazing, Brewster ~56°: Fresnel equations (Wikipedia);
  Cornell graphics Fresnel note.
- Low-E / coated glass NIR reflectance ≥20% (–80%) at 905 nm: AccuCoat LiDAR coatings; *Designing
  LiDAR-Detectable Dark-Tone Materials …*, Adv. Funct. Mater. 2025 (10.1002/adfm.202414876);
  US patent 11,977,154 (NIR detection coatings).
- LiDAR range equation / detection vs reflectance, angle, noise floor (1/R²–1/R⁴, max range ∝ √ρ):
  SPIE *LiDAR Range Equation*; calibvision *LiDAR Detection Range vs Reflectance*; LeddarTech.
- Oren–Nayar rough-diffuse / Cook–Torrance microfacet (the g and b/m terms): standard BRDF texts.

---

## 9. Corrections from user notes + return-probability done right (2026-06-08, round 2)
Two physical facts the user supplied, tested against data (`analysis/reexamine.py`,
`analysis/return_prob2.py`):

**(a) Bech has BOTH dials.** Bech glass is **+48 brighter head-on (0–2°) than its 8–14° floor** —
a real **near-normal flash** sitting under the strong **grazing Fresnel rise**. (L6 flash +119,
Oxy +22.) So Bech = small head-on flash + dark absorptive floor + big grazing Fresnel. Earlier text
that called Bech "flash-less" was wrong.

**(b) Oxy had curtains behind the glass → "Oxy glass" was mostly the curtain.** In-window depth is
**bimodal**: a small front layer (~0, 13 %) and a dominant peak at **~0.30 m (71 %)** — the curtain.
The dense, flat, range-independent ~140 signal I characterized is the **matte curtain seen through
clear/transmissive glass**, not a coated glass surface. **Retract "Oxy = coated diffuse, L6-like."**
Oxy is the realistic "transmissive glass with a backing behind it" case (good for training — many
real windows have blinds/curtains/interior behind).

**Return vs no-return probability — measured with the beam pattern divided out:**
- The L2 is pitched so its spin axis points at the wall, so **beam density per steradian falls ~20×
  from boresight (φ≈2.5°) to φ≈40°** (measured; the dashed curves in `return_prob_corrected.png`).
  My earlier raw "return density vs angle" was mostly *this*, not the glass. **Crucially the sim
  already reproduces this** (it fires the same L2 pattern), so the forward model must NOT re-apply
  n(φ) — it only supplies the per-beam P_return(material, θ, R).
- After dividing n(φ) out (window returns/sr ÷ opaque-facade returns/sr at matched φ, frontal scans
  where φ=incidence): **through-window P_return is order-1 and roughly flat** for Oxy/L6 — something
  behind the glass (curtain / coating / interior) almost always sends a beam back, so "no return" is
  rare. For Bech it is lower and noisier (dark, transmits, depends on what's inside). The estimate is
  noisy because these panels are **almost all window** → little opaque facade to reference.
- **The absolute noise floor T is NOT recoverable from these bags.** Bags record only returns (no
  missed-beam flag; cloud is dense, no zeros). The faintest *detected* intensity (~30 counts) does
  **not** rise as R² with range — it *falls* — i.e. T sits below everything we ever detect, so we
  never sample the threshold. To get T one needs: a controlled **dark-target-walked-back** test
  (range where returns vanish → T/C at that reflectance), the **L2 datasheet** range-vs-reflectivity
  spec, or a sensor mode that emits no-return flags. **For the model:** set T low and treat the
  return gate mainly as the **angular/material geometry** (specular-away vs diffuse/coated/backing
  returns) — the 1/R² threshold only bites for **dark + far** targets.

**Equations (the brightness model, explicit).** Reported intensity (range-compensated) for a beam at
incidence θ, per material (the three dials of §7):
```
I(θ) = a·((1−g)·cosθ + g)            # matte floor  (g: 0=cosine fade, 1=perfectly flat)
     + s·F(θ; n≈1.5)                 # grazing rise (F = unpolarized Fresnel, 0→1 toward 90°)
     + b·exp(−tan²θ / m²) / cos⁵θ     # head-on flash (Beckmann lobe, width m = roughness)
I_obs = min(I(θ), 255)               # sensor clips (saturates) at 255
```
Detection (separate channel; this is the ONLY place range lives):
```
returns iff  C · I(θ) / R²  >  T      # received power over the noise floor T
P_return(θ) ≈ p_floor + (1−p_floor)·exp(−θ²/2cone²)   # geometry gate: p_floor≈1 diffuse/backing, ~0 mirror
```
Fitted dials (counts, C folded in): L6 `a≈123,g≈.28,s≈256,b≈139,m≈6°`; Oxy(curtain) `a≈142,g≈.59,
s≈355,b≈22`; Bech `dark floor a≈100, big s, small flash b≈78` (and high transmission → interior).

---

## Method & caveats
- **Roll-invariance**: on a frontal (or any) facade, incidence angle (vs the surface normal),
  range, depth-behind-plane, and intensity are invariant to the boresight roll about x. So the
  physics needs the plane right (verified <1°), not α exact; α only affects window-vs-frame label
  assignment near edges. (`analysis/seg.py`.)
- Material separation is **GT-geometry-only** (window apertures), never intensity — so intensity is
  independent evidence. The huge Bech glass/frame contrast (Δ≈104, clean depth split) confirms the
  apertures register correctly.
- **Incidence angle** = angle(beam, fitted-facade normal); exact normals from GT/fit.
- **Return-RATE vs angle** is still not directly measured (bags hold only returns, no nominal beam
  grid). Qualitatively all glasses populate the apertures densely at all angles (Bech via interior
  at low angle + surface at grazing); the old hard "specular cone gate" stays falsified.
- GT mullions are hairline (Oxy/Bech panels are abutting windows) → the "frame" class is tiny there;
  the panels are effectively all-glass + ground for labeling.
- Tooling: `analysis/{explore_raw,facade_view,seg,agg,probe_alpha,catalog,physics_dash,angular_fit}.py`.

---

# Appendix A — L6 glass: detailed behavior


A data-derived characterization of one architectural glass type as seen by a **Unitree L2
LiDAR (~905 nm)**, to inform the first-principles forward model. Built from **43 real test-bed
scans** (4052 frames, ~10.8 M labeled glass points) of a 2×4 multi-window panel swept over
standoff (1.1–1.8 m) and incidence (yaw/pitch/roll up to ±30°). **Treat the current model as a
hypothesis; the data below falsifies parts of it.**

## TL;DR
This glass is **visibly see-through but NIR-reflective and glossy-to-diffuse**, NOT a specular
mirror. It returns the LiDAR beam **densely at all incidence angles (to ~76°)**, with a sharp
near-normal peak on top of a **near-flat off-normal floor**. The current model's **glass cone
gate is wrong for it**, its **Lambertian `a·cosθ` diffuse term decays too fast**, and its
**`1/R²` range law does not describe the reported intensity** (the L2 outputs a range-compensated
reflectivity). A *different* glass scanned later ("black in sun") behaves specularly (narrow
forward cone only) — so glass is a **spectrum by type/coating**, not one model.

## 1. Return vs incidence angle (the big one)
θ = angle between beam and the glass-plane normal (normal known exactly from test-bed GT).
- Glass returns exist at **all θ**: median **27°**, p90 **53°**, **max 76°**.
- Fraction of glass returns beyond the model's glass cones:
  θ>3°: **99%**, θ>9° (widest prior cone, low-E): **92%**, θ>30°: **43%**, θ>45°: **19%**, θ>60°: **4%**.
- => A near-normal "specular cone" (prior half-widths 3–9°) would discard **~92%** of the real glass
  returns. **The cone-gate assumption is falsified for this glass.** It behaves like a *diffuse*
  surface (uniform-ish across angle), the opposite of a whiteboard-style specular reflector.

## 2. Intensity vs incidence angle  I(θ)
Reported L2 intensity is already range-compensated (see §3), so reported intensity ≈ ρ_eff(θ).
Binned median (L2 counts, peak ≈ 255 at normal):

| θ (deg) | 1.5 | 7.5 | 13.5 | 22.5 | 31.5 | 40.5 | 52.5 | 64.5 |
|--------|-----|-----|------|------|------|------|------|------|
| median I | 255 | 152 | 126 | 130 | 80 | 108 | 91 | 98 |

Shape = **sharp near-normal peak** (255→~150 within ~8°) sitting on a **broad, nearly FLAT floor
(~90–130 counts) that persists to 65°+** and does *not* fall off like cosθ.

**Fit of the model's own form** `I = a·cosθ + b·exp(−tan²θ/m²)/cos⁵θ` (counts, C=1):
- `a (diffuse) ≈ 129`, `b (specular) ≈ 134`, `m ≈ 0.10 rad (5.8°)`; diffuse fraction at normal **≈ 0.49**.
- **R² ≈ 0.78** — captures the peak+floor but **misfits the floor's angular shape**: real floor is
  ~flat/rising at grazing, the `a·cosθ` term decays, so the model under-predicts θ>40°.
- Compare the prior "clear glass": `a=0.01, b=0.05, m=1.7°, cone=2.9°` (a near-delta spike, diffuse
  fraction 0.17) — qualitatively nothing like this glass.

## 3. Intensity vs range — the L2 reports range-compensated reflectivity (n≈0)
Median glass intensity at near-normal, by standoff group:

| standoff | median R | median I |
|----------|----------|----------|
| ~1.65 m | 1.80 m | 147 |
| ~1.51 m | 1.68 m | 145 |
| ~1.40 m | 1.57 m | 141 |
| ~1.11 m | 1.28 m | 136 |

Raw `1/R²` power would change ~2× over this range; the data is **flat (≈±4%)**. Log-log slope
n≈0. => **The L2's reported "intensity" is a range-normalized reflectivity, not raw received
power.** The model's `P_r = C·ρ_eff/R²` (n=2) therefore does **not** describe the reported
intensity value. (The physical `1/R²` still governs whether a faint beam clears the noise floor —
i.e. the *return probability* — but that is a separate quantity from the reported intensity.)

## 4. Material contrast (sanity / labeling)
- Glass (in-window) median intensity ≈ **120–150**; metal frame/mullion (between windows) ≈
  **150–177** — frame is consistently **brighter** (Δ +17…+55). Useful cross-domain cue.
- The window grid registers correctly: glass clusters measure 1.54–1.60 × 1.11–1.13 m vs GT
  1.63 × 1.15 m.
- Depth note (benign): the panel sits ~13–18 cm *behind* the GT plane — an accumulated facade
  measurement offset (constant across bags; verified rotation-invariant, i.e. not a transform
  artifact). A small, spatially-localized **deep see-through tail** (>~30 cm behind) exists only
  where something is actually behind a pane ("interior" label).

## 5. Physical interpretation
- **Active sensor:** the L2 uses its own ~905 nm laser; **sunlight does not drive the returns**.
  The visual look (see-through in sun / opaque-bright near sunset) reports the coating's *visible*
  reflectance, which only *correlates* with NIR.
- The combination **visible-transparent + strong, glossy NIR return** is the signature of a
  **coated glass (low-E / solar-control)**: published Ag/ITO low-E stacks give **50–80% NIR
  reflectance at ~80% visible transmission**. The coating (plus surface micro-roughness/contam.)
  yields a **glossy BRDF**: a specular lobe near normal + a substantial off-normal floor. The
  floor being flat/rising at grazing is consistent with **Fresnel reflectance rising toward grazing**
  combined with enough roughness to scatter some of it back to the monostatic detector.
- This is the **opposite** of the textbook "clear glass is invisible to LiDAR (specular-only,
  see-through)". This glass is LiDAR-*friendly* (dense returns) — good for detection, but it means
  the specular-cone model does not apply.
- A second glass type scanned later ("black in sun"; returns only in a **narrow forward cone**,
  nothing off-cone) is the **specular** end — that one IS what the cone gate models (a tinted/
  absorptive glass acting as a NIR mirror).

## 6. What this means for the forward model (recommended changes)
The model's **functional form (Lambertian + Beckmann) is the right general framework** (it is the
accepted "glossy-to-rough" model), but three structural assumptions must change:

1. **Cone gate must be per-type, not "all glass".** For this (diffuse/coated) glass, disable the
   cone (`is_glass=False`, or cone≈90°): returns come at all θ. Keep the narrow cone only for
   specular types (the "black" glass). Better: make the gate a soft return-probability vs θ, fit
   per glass type, rather than a hard cutoff.
2. **Fix the diffuse angular term.** Real off-normal floor is ~flat/rising at grazing, not `cosθ`.
   Options: add an angle-independent (or Fresnel-grazing) component, or use a heavier-tailed lobe.
   The pure `a·cosθ` under-predicts grazing glass returns.
3. **Separate "reported intensity" from "received power".** The L2 reports a range-compensated
   reflectivity (n≈0). Calibrate intensity against ρ_eff(θ) directly; apply `1/R²` only inside the
   return-probability/threshold stage (raw power vs noise floor), not to the reported value.
4. **Glass is a spectrum by type/coating** (clear/tinted/low-E/solar): one fixed glass model can't
   span specular-cone ↔ glossy-diffuse. Fit per scanned type; carry the type as a parameter.

**Fitted starting point for THIS glass** (L2 counts, range-compensated, C=1):
`a≈129, b≈134, m≈0.10 rad, cone=OFF, n_intensity≈0` — but expect to replace the cosθ floor with a
flatter/grazing term (current R²≈0.78).

## 7. Open items / caveats
- Need the "black in sun" specular glass scans to fit the specular-cone branch and confirm the
  two-type spectrum.
- The flat/rising grazing floor deserves a dedicated BRDF fit (Fresnel + roughness) rather than the
  ad-hoc Lambertian+Beckmann; current fit R²≈0.78.
- Return-RATE vs θ (vs the L2 nominal beam grid) not yet computed — only returns are in the bags.
  Needed to fit a soft return-probability(θ) instead of a hard cone.
- Intensity is in raw L2 counts (≈0–255), not absolute reflectance; `a,b` carry that scale.

## Sources
- LiDAR vs glass (transmit + specular + diffuse; specular returns only near normal, diffuse uniform):
  autonomoussystems.net LiDAR reflectivity (ICRA 2012); "Detection and Utilization of Reflections in
  LiDAR Scans" (PMC11314935).
- Low-E / coated glass NIR reflectance (50–80% NIR @ ~80% visible transmission): "Materials for
  reflective coatings of window glass applications" (ResearchGate 257388739); "Designing
  LiDAR-Detectable Dark-Tone Materials with High NIR Reflectivity" (Adv. Funct. Mater. 2025,
  10.1002/adfm.202414876).
- Lambertian+Beckmann as the glossy-to-rough model; range+incidence radiometric normalization:
  "Radiometric Calibration ... Hyperspectral LiDAR Backscatter Intensity" (MDPI RS 12(17):2855);
  "Analysis and Radiometric Calibration ... Incident Angle Effect" (Sensors 21:2960);
  ISPRS "Normalization of LiDAR Intensity Data Based on Range and Surface Incidence Angle".

---

# Appendix B — forward-model glass diversification


**Goal:** turn the falsified single-glass model into a **per-type** glass model, predict parameters
for the common architectural glass types, and **domain-randomize** over them in sim so a PointNet++
trained on the synthetic set generalizes to real glass we can't all physically scan.

Status: the data (`GLASS_BEHAVIOR_L6.md`) + the model-AI agree on the revision below. **Only the
low-E/coated row is data-fitted; the rest are physics+literature PREDICTIONS** (clearly marked).

---

## 1. Revised model (per point, per glass type)

Two **separate** channels (the old model wrongly fused them):

**(A) Reported intensity** — range-independent (L2 outputs a range-compensated reflectivity, n≈0):
```
I(θ) = floor(θ) + b · exp(−tan²θ / m²) / cos⁵θ            # specular lobe on a diffuse floor
floor(θ) = a + g · F(θ)                                    # NOT a·cosθ : flat base + grazing rise
F(θ) = unpolarized Fresnel reflectance(θ, n_glass≈1.5)     # ~0.04 at 0°, →1 toward 90°
```
- `a`  = flat diffuse floor (the ~constant off-normal level we measured)
- `g`  = grazing-rise weight (Fresnel climb toward grazing — explains the flat/rising floor)
- `b`  = near-normal specular peak; `m` = lobe width (roughness). a,b INDEPENDENT.

**(B) Return probability** — whether a beam comes back at all (this is where 1/R² lives):
```
P_return(θ) = P_floor + (1 − P_floor) · exp(−θ² / cone²)   # soft, replaces the hard cone gate
keep  iff  P_return(θ) high enough  AND  raw_power = C·I(θ)/R² > T
```
- `P_floor` = off-normal return floor: **1.0 = diffuse (returns everywhere), ~0 = specular (cone only)**.
  This single knob spans the whole specular↔diffuse spectrum; the old hard cone is just `P_floor=0`.
- `cone` = near-normal width (only matters when `P_floor` is low).

**(C) Transmission** `τ` — fraction of beams that pass through the glass (→ a return from whatever is
behind, or none). Drives "see-through"/interior. Clear glass high τ; coated/tinted low τ.

So each glass TYPE = `{a, g, b, m, P_floor, cone, τ}`. The old model = this with `g=0`, `P_floor=0`.

---

## 2. Predicted parameters per glass archetype

Normalized (peak diffuse-white ≈ 1.0; scale to sensor counts via C). **conf** = confidence.

| type | a (floor) | g (grazing) | b (spec) | m (deg) | P_floor | cone(deg) | τ (transmit) | conf |
|------|-----------|-------------|----------|---------|---------|-----------|--------------|------|
| **clear / uncoated float** | 0.02–0.08 | 0.3–0.6 | 0.10–0.30 | 1–3 | 0.02–0.10 | 2–5 | 0.70–0.90 | pred (lit) |
| **tinted / body-absorptive ("black-in-sun")** | 0.02–0.06 | 0.2–0.5 | 0.10–0.30 | 2–6 | 0.05–0.20 | 3–8 | 0.20–0.50 | semi (your obs) |
| **low-E / solar-coated  ← OUR GLASS** | 0.40–0.55 | 0.3–0.7 | 0.45–0.55 | 4–9 | 0.85–1.00 | n/a | 0.10–0.40 | **DATA-FIT** |
| **reflective / mirror (heavy coat)** | 0.10–0.30 | 0.4–0.8 | 0.55–0.85 | 2–6 | 0.40–0.80 | 5–15 | 0.00–0.20 | pred (lit) |
| **frosted / etched / patterned** | 0.40–0.70 | 0.2–0.5 | 0.00–0.15 | 15–40 | 0.90–1.00 | n/a | 0.10–0.40 | pred |
| **spandrel / opacified (opaque)** | 0.35–0.60 | 0.1–0.3 | 0.00–0.10 | 20–40 | 1.00 | n/a | 0.00 | pred (~wall) |

Anchor: our low-E fit was `a≈129, b≈134, m≈5.8°` in L2 counts (peak≈255) → `a≈b≈0.5`, `m≈6°`,
`P_floor≈1` (returns to 76°), with the flat/rising floor captured by `g`. The other rows place each
type on the spectrum from that anchor using the physics (Fresnel, coating NIR reflectance 50–80%,
tint absorption, roughness) and the literature in `GLASS_BEHAVIOR_L6.md`.

---

## 3. Diversification (domain randomization) for sim training

Per window (or per building) in the generator:
1. **Sample a glass type** from a weighted prevalence (e.g. clear 0.30, low-E 0.30, tinted 0.15,
   reflective 0.10, frosted 0.10, spandrel 0.05 — tune to your target region/GCC).
2. **Sample `{a,g,b,m,P_floor,cone,τ}` uniformly within that type's ranges** (ranges above; widen
   ~20% beyond to cover sim-to-real gap).
3. Apply the revised forward model with those params: paint intensity `I(θ)`, decide returns via
   `P_return(θ)` + range threshold, and for transmitted beams (prob τ) return off whatever is behind
   (room/sky/backing) or drop.
4. Label as `glass` regardless of type (the cleaning target); keep the fine type id in the raw npz
   for analysis. Frame/wall/ground unchanged.

Train PointNet++ on the **union**. Because the set spans specular↔glossy↔diffuse and high↔low
transmission, the classifier learns "glass-ness" as a *distribution of behaviors*, not one signature
— so it should hold up on real glass types we never scanned.

**Why this is the right call:** we can only data-anchor one type; randomizing over a physically
bounded spectrum is the standard sim-to-real generalization method. **Risk to manage:** if real glass
falls outside the sampled ranges, generalization fails → keep ranges generous and physically bounded,
and **anchor more types whenever you can scan them** (each real scan replaces a predicted row with a
fitted one). The "black-in-sun" scans are the highest-value next anchor (the specular branch).

---

## 4. Implementation outline (on approval)

1. `forward_model/intensity.py`: add the Fresnel-grazing floor `a + g·F(θ)` (keep old `a·cosθ` path
   for back-compat via `g=0`).
2. `forward_model/returns.py`: replace hard `cone_gate` with soft `P_return(θ)=P_floor+(1−P_floor)·gauss`.
3. `forward_model/range_model.py` / `apply.py`: split **reported intensity** (no 1/R²) from
   **return-probability** (1/R² vs threshold). `MaterialParams` gains `g, P_floor` (cone kept).
4. New `forward_model/glass_types.py`: the archetype table + a sampler `sample_glass(rng, region)`.
5. Generator hook (`buildings/`): assign a sampled glass type/params per window.
6. Tests (TDD): each archetype returns the expected angular signature (specular→cone-only,
   low-E→all-angles, etc.); back-compat (g=0,P_floor=0) reproduces current behavior.

## 5. Open items
- **Return-rate vs θ** against the L2 nominal beam grid → fits `P_floor/cone` properly (bags hold
  only returns). Highest-value missing measurement for the return channel.
- Scan the **"black-in-sun"** glass → anchor the specular row.
- Dedicated **Fresnel+roughness BRDF** fit for the floor (current Lambertian+Beckmann R²≈0.78).
