# Images YOU need to capture (can't be generated from data)

Save each into `presentation/figures/` with the exact filename below, then upload it to
Claude Design alongside the generated figures. Each entry says what the shot must show and
which slide it lands on.

---

### 1. `S14_ue5_sim.png` — the simulator (slide 14, "sim environment")
**What:** a CosysAirSim / UE5 viewport screenshot showing one of our generated buildings with
the drone + LiDAR, in **ComputerVision mode**.
**How:** start UE5 in Play with `settings_cv_stdlidar.json` active, then
`python3 -m buildings.ros_publish --square-prob 0.6 --region gcc`, and screenshot the UE5
window (ideally with the building facade filling the frame). A clean third-person view of the
drone in front of a glass facade is perfect.
**Why it matters:** this is the "where the synthetic data comes from" visual — graders want to
see the sim, not just plots.

### 2. `S07_testbed_photo.png` — the physical rig (slide 7, "physical data gathering")
**What:** a photo of the real 6-DOF test-bed (the CNC-like rig) with the Unitree L2 mounted,
pointed at a glass facade.
**How:** phone photo, landscape, rig + facade both in frame. If you have a shot mid-scan,
even better.
**Why it matters:** pairs with the schematic `S07_testbed_schematic.png` — schematic explains
it, photo proves it's real hardware.

### 3. `S08_rviz_scan.png` — RViz labelled scan (slide 8 or 10) — *optional*
**What:** RViz screenshot of a real labelled scan, points colored by class
(blue glass / gold frame / green ground / red interior).
**How:** `python3 -m calibration.rviz_publish --bags 1,7,20` then load `calibration/labeled.rviz`;
screenshot the RViz 3-D view. (A matplotlib equivalent — `S08_pointcloud_explainer.png` /
`S10a_labels.png` — is already generated, so this is a "nicer-looking system view" upgrade, not
required.)
**Why it matters:** an RViz capture reads as "this is a working ROS2 perception system."

### 4. `S12_facade_photos.png` — the three real facades (slide 12) — *optional*
**What:** phone photos of the three buildings you scanned — **L6**, **Oxy**, **Bechtel** — side
by side (one composite, or upload three and I'll note placement).
**How:** straight-on photos of each glass facade.
**Why it matters:** grounds the glass-physics EDA in reality — "these are the three glass types
we measured," next to the intensity-vs-angle curves.

---

**Tip:** keep all filenames exactly as above. In `DECK_PROMPT.md` each of these is referenced
on its slide with an `IMAGE:` line, so matching names lets Claude Design place them correctly.
If you can't get one of the optional shots, just skip it — the slide still works on the
generated figures alone.
