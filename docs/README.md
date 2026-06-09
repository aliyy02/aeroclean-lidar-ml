# Docs index

- **[calibration.md](calibration.md)** — how real Unitree-L2 scans are labelled against
  ground-truth window corners (axis mapping, boresight roll, glass-plane fit, perpendicular
  projection + region-grow, the four classes).
- **[glass_characterization.md](glass_characterization.md)** — the glass EDA: how the three
  glass types (L6 coated-specular, Oxy coated-diffuse, Bechtel tinted-Fresnel) behave vs
  incidence angle, saturation at 255, and see-through interior returns. The forward-model
  verdict + calibration plan. (Appendices: L6 detail, model diversification.)
- **window_types.pdf** — facade/window priors used by the building generator.
- **lidar_post_processing.pdf** — reference notes on the forward model / post-processing.

Package-level docs:
- `../forward_model/README.md` — the per-beam physics model.
- `../calibration/README.md` — the labelling package.
- `../buildings/DIVERSITY_SPEC.md` — procedural building diversity spec.
