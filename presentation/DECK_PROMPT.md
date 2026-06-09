# Claude Design brief — AeroClean LiDAR perception (EECE 490)

> **How to use this file.** Open Claude Design. **First upload every PNG** from
> `presentation/figures/` (and any photos you captured from `CAPTURE_LIST.md`). Then paste
> this whole file as the prompt. Each slide has an `IMAGE:` line naming the figure to place
> and describing it, so Claude can place it even if it can't match the filename. After it
> builds the deck, do one quick pass in the live canvas to fix any mis-placed image.

---

## Meta-instructions for Claude Design (read before building)

**Build a ~18-slide technical presentation** titled
**"AeroClean: Learning to See Glass — LiDAR Perception for an Autonomous Facade-Cleaning Drone."**

- **Audience:** EECE 490 (Applied Machine Learning) instructor + classmates. Technically literate
  but NOT LiDAR specialists.
- **Tone & style:** clean, modern, **visual-first**. Lead with the figure on each slide; keep
  text to short bullets. Explain **concepts and a little math**, almost **no code**. **No
  unexplained jargon** — if a term like "incidence angle," "Fresnel," or "set abstraction" is
  used, give a one-line plain-English gloss. Confident but honest: clearly mark what is built
  vs. what is pending.
- **Visual system:** consistent palette throughout. Where class colors appear, use
  **blue = glass, gold = frame, green = ground, red = interior**. One accent color for
  headers. Big readable type. Diagrams over paragraphs.
- **Narrative spine (do not reorder):** the problem → the geometric baseline and why it breaks →
  what the literature offers and why it doesn't fit → why a learned model (PointNet++) → how we
  got real data (the test bed, what a point cloud is) → reframing the task as segmentation →
  labeling with ground truth → simulation + the sim-to-real gap → an EDA showing how brutal
  glass is for LiDAR → the physics forward model that closes the gap → the sim environment →
  why synthetic training is future work → the model/results (pending) → responsible ML →
  evaluation + the research question.

Each slide below = one slide. `[TRUE]` = established/measured this project; `[PLACEHOLDER]` =
not yet measured, leave as clearly-pending.

---

## Slide 1 — Title  `[TRUE]`
- **AeroClean: Learning to See Glass**
- Subtitle: LiDAR perception for an autonomous window-cleaning drone
- Course/team line: EECE 490 · Ali Yaakoub, Ali Nasrallah, Mohammad Kanaan
- IMAGE: (optional) `S14_buildings_gallery.png` or a clean drone/facade render as a faint
  background; keep the title dominant.
- Speaker note: one sentence — "we're building the eyes for a drone that cleans skyscraper glass."

## Slide 2 — The problem & why it matters  `[TRUE]`
- High-rise window cleaning is **dangerous, slow, and expensive** — done by humans on ropes and
  cradles; a multi-billion-dollar global market.
- AeroClean = an autonomous drone that cleans glass facades. To clean a pane it must know
  **where each window is to within a few centimetres**, relative to the building.
- The hard part isn't flying — it's **perceiving the facade**, and facades are mostly **glass**,
  which is nearly invisible to most sensors.
- Speaker note: set up that perception (not control) is the bottleneck this talk is about.

## Slide 3 — First approach: a geometric pipeline  `[TRUE]`
- We first solved a simpler version: detect **one flat board** with a hand-built **16-stage
  geometric pipeline** (RANSAC plane fit → project to 2-D → multi-frame smoothing → boundary
  trace → PCA corner walk).
- It **works well on a single board** in simulation: clean plane, four corners, stable.
- IMAGE: `S03_ransac_baseline.png` — LEFT panel; a single flat board collapses to one tight
  RANSAC plane.
- Speaker note: "RANSAC = repeatedly guess a plane, keep the one most points agree with."

## Slide 4 — …and why it breaks on real windows  `[TRUE]`
- A real facade is **not one plane**: the frame sits in front, the glass behind it, and beams go
  **through** the glass and bounce back from **inside the room** — several depth layers at once.
- The 16-stage pipeline assumes **one plane, one board, head-on** — it breaks on multi-window,
  glass, and oblique views. It's also **open-loop** and hand-tuned.
- IMAGE: `S03_ransac_baseline.png` — RIGHT panel; the same scene shows MANY depth layers
  (frame / glass / interior) a single plane can't represent.
- Speaker note: this motivates everything that follows.

## Slide 5 — What the literature offers (and why it doesn't fit)  `[TRUE]`
- Point-cloud segmentation is a mature field — but the standard models/datasets (e.g. S3DIS,
  ScanNet) are **indoor RGB-D scans of furniture**, not **905 nm LiDAR looking at glass curtain
  walls outdoors**.
- Glass returns, saturation, see-through interior points, and our sparse spinning-LiDAR geometry
  are **out of distribution** for off-the-shelf models.
- Takeaway: the **method** (deep point-cloud segmentation) transfers; the **data and physics** do
  not — so we must build our own.
- Speaker note: keep it short; one or two cited examples as bullets.

## Slide 6 — Why a learned model — and why PointNet++  `[TRUE]`
- RANSAC leaned on two cues: **local point density** and **local geometric structure**. It
  encoded them as **hand-tuned thresholds** that glass violates.
- **PointNet++ learns those same cues automatically**: it samples **local neighbourhoods**
  ("query balls"), pools them into **hierarchical features**, and works **directly on unordered
  points** (no grid, no voxels).
- So we keep the signal RANSAC used, but make it **learned and robust** to glass / multi-window /
  oblique.
- IMAGE: `S06_pointnet_motivation.png` — query balls over a real scan + the "why PointNet++"
  panel.
- Speaker note: PointNet → PointNet++ in one line (adds local hierarchy).

## Slide 7 — Getting real data: a 6-DOF LiDAR test bed  `[TRUE]`
- To train and validate we need **ground truth**. We built a **6-DOF, CNC-like rig** that holds
  the **Unitree L2** LiDAR and moves it to **precisely known poses** in front of a glass facade.
- Every scan therefore comes with a **known sensor position** and **known window-corner
  geometry** — exactly what supervised learning needs.
- IMAGE: `S07_testbed_schematic.png` (the diagram) + `S07_testbed_photo.png` (your real rig
  photo) side by side.
- Speaker note: "6-DOF = position + orientation fully controlled."

## Slide 8 — What a point cloud actually is  `[TRUE]`
- Every laser beam that returns gives **one point**: position **(x, y, z)** plus a **return
  intensity (0–255)**.
- A scan is **thousands of these points** — **unordered, no grid, no pixels**. Density varies
  with range and surface.
- Glass is the catch: it mostly lets the beam **through**, so it returns **few, erratic** points.
- IMAGE: `S08_pointcloud_explainer.png` — one real frame colored by intensity + the explainer
  panel. *(If you captured `S08_rviz_scan.png`, use it here instead for a "live system" look.)*
- Speaker note: this is the slide that teaches the audience to read every later plot.

## Slide 9 — Reframing the task: segment, then extract corners  `[TRUE]`
- New plan: don't fit geometry blindly. **Classify every point** into
  **glass / frame / interior / ground**.
- Then run a **light geometric step on the glass points only** — sort the boundary, fit a
  quad, read off the four corners (reusing the corner logic from the old pipeline, but now on
  clean glass-only points).
- Splitting "**what is it?**" (learned) from "**where are the corners?**" (geometry) makes each
  part simple and robust.
- IMAGE: a simple 3-box flow diagram — *LiDAR scan → PointNet++ segmentation (glass/frame/
  interior/ground) → corner extraction on glass*. (Ask Claude to draw this; or reuse
  `S10a_labels.png` as the middle stage.)
- Speaker note: corner extraction is implemented after inference (near-term work).

## Slide 10 — Labeling with ground truth  `[TRUE]`
- Using the rig's known window corners, we label every point: in-pane on the glass plane =
  **glass**; between/around panes = **frame**; behind the glass = **interior**; floor =
  **ground**.
- Real tape-measured ground truth has **centimetre-scale global error**, so we **fit the true
  glass plane** and project points **perpendicularly** onto it, then **region-grow** to fix
  recessed frames — getting clean labels despite the measurement error.
- **142 real scans labelled** across **three buildings** (L6: 43, Oxy: 59, Bechtel: 40).
- IMAGE: `S10a_labels.png` (labelled scans, all three buildings) + `S10b_class_distribution.png`
  (class mix — glass is the minority class).
- Speaker note: emphasise this was the hard, multi-day data-engineering effort.

## Slide 11 — Simulation and the sim-to-real gap  `[TRUE]`
- We also generate data in **simulation (CosysAirSim on UE5)** — it gives **perfect geometry and
  free labels**, infinite buildings, no tape measure.
- But the simulator's LiDAR is **too clean**: every beam returns, glass is solid, intensity is
  uniform. Real glass is **sparse, saturated, and see-through**.
- That mismatch — the **sim-to-real gap** — is what we have to close before synthetic data can
  train a model that works on real glass.
- IMAGE: `S11_sim_to_real_gap.png` — simulator (dense, uniform) vs real glass (sparse,
  saturated).
- Speaker note: this sets up the EDA and the forward model.

## Slide 12 — EDA: glass is brutal for LiDAR  `[TRUE]`
- We measured how the **three glass types** actually behave:
  - **L6 (coated/specular):** a bright **spike head-on**, then dim.
  - **Oxy (coated/diffuse):** **flat and bright** at all angles.
  - **Bechtel (tinted):** dim head-on, **rising to full saturation (255) at grazing angles**
    (Fresnel reflection), and **see-through** — beams return from up to 2.5 m **inside the room**.
- Two universal effects: **saturation** (the sensor clips at 255) and **transmission** (interior
  returns behind the glass).
- IMAGE: `S12a_angular_glass.png` (intensity vs incidence angle, 3 types), with
  `S12b_saturation_hist.png` (255 clipping) and `S12c_seethrough.png` (beams returning from
  inside the room) as supporting panels. *(Optional: add `S12_facade_photos.png` of the three
  buildings.)*
- Speaker note: the centrepiece EDA slide — "this is why naïve geometry fails."

## Slide 13 — A first-principles forward model  `[TRUE]`
- To make simulated glass look real, we built a **physics forward model** that, per beam,
  computes:
  - **Reflectance ρ(θ)** — how bright the return is vs **incidence angle θ** (a diffuse term +
    a **grazing-angle (Fresnel) lift** + a near-normal **specular burst**). This is
    **range-independent**.
  - **Return probability P(θ)** — *does the beam come back at all?* (matte coatings return
    always; mirror-like glass only near head-on.)
  - plus **saturation clipping at 255** and **transmission → interior returns**.
- Range only enters once, as a **1/R² noise floor** (faint far returns are lost) — matching the
  real sensor.
- IMAGE: `S13_forward_model.png` — reflectance ρ(θ) and return-probability P(θ) for the three
  glass archetypes.
- Speaker note: "first-principles" = derived from optics, then calibrated to our real scans.

## Slide 14 — The simulation environment  `[TRUE]`
- **CosysAirSim + Unreal Engine 5**, using the **Standard LiDAR in ComputerVision mode**: it
  gives **clean geometry + free per-point ground-truth labels** (which object each beam hit) but
  **no intensity** — so the forward model supplies the optics.
- A **procedural building generator** makes endless varied facades (multiple **facade systems**
  and **square / L / U / T footprints**) to train for generalisation.
- IMAGE: `S14_ue5_sim.png` (your UE5 screenshot) + `S14_buildings_gallery.png` (the procedural
  gallery).
- Speaker note: labels are free in sim; realism comes from the forward model.

## Slide 15 — Why synthetic training is (deliberately) future work  `[TRUE]`
- The forward model can only be **trusted once calibrated**, and calibration needed the **real
  labelled scans** — which took **days** to get right (tape-error correction, sensor-axis and
  boresight discovery, three glass types).
- So far we trained on the **real** labelled data; **synthetic training is next** — it's what
  buys **generalisation** to glass we've never scanned. We'll validate the physics model against
  **other campus facades** before relying on it.
- Honest framing: a sequenced plan, not a gap.
- Speaker note: this pre-empts "why didn't you train on sim yet?"

## Slide 16 — Model & results  `[PLACEHOLDER]`
- **Model:** PointNet++ semantic segmentation — set-abstraction + feature-propagation layers,
  ~16k points per scan, 4 classes. Trained on the labelled real data; served via a
  **dockerized inference API** (Colab GPU).
- **Results:** *pending inference.* Leave the metrics clearly marked TO-FILL.
- IMAGE: `S16_results_placeholder.png` — confusion-matrix + metrics template marked PENDING
  INFERENCE.
- Speaker note: state plainly that inference + corner extraction are the immediate next step;
  do **not** invent numbers.

## Slide 17 — Responsible ML  `[TRUE]`
- **Distribution shift:** trained on three buildings + sim; new glass/lighting is out of
  distribution → mitigations: synthetic diversity, calibration checks on new facades.
- **Failure & safety:** a mis-detected corner near a glass wall is a **physical** risk →
  geometric sanity checks, confidence thresholds, conservative fallback behaviour.
- **Explainability:** per-point class + geometric corner step is **inspectable** (you can see
  why), unlike an end-to-end black box.
- **Data bias:** glass is the **minority class** and our three buildings under-represent global
  glass diversity → class weighting + more data.
- Speaker note: rubric needs ≥3 responsible-ML topics; this covers four.

## Slide 18 — Evaluation plan & the research question  `[TRUE]`
- **Research question:** *Can a learned point-cloud segmenter, trained largely on physics-based
  synthetic LiDAR, detect real glass-facade windows accurately enough (corners within a few cm)
  for autonomous cleaning?*
- **How we'll measure it:** segmentation **mIoU / per-class F1** (esp. glass), **corner error
  in cm**, **inference rate on the Raspberry Pi 5**, and the **sim→real transfer gap**.
- **Next steps:** run inference, extract corners, then validate the forward model on new
  facades.
- Close with a thank-you / questions line.

---

## Rubric mapping (for your check — not a slide)
- **Type A (application of ML to a real problem):** entire deck.
- **Data engineering / EDA:** slides 7–13.
- **Method justification:** slides 5–6.
- **Responsible ML (≥3 topics required):** slide 17 (4 topics).
- **Evaluation:** slides 16, 18.
- **Honesty about limitations:** slides 11, 15, 16.

## What is still a placeholder (do not overstate)
- All **model accuracy numbers** (slide 16) — pending the friend's inference run.
- Corner-extraction-from-segmentation is **designed, not yet implemented**.
- The forward model is **calibrated on three buildings**; cross-facade validation is future work.
