# Supplementary-figure gap audit

Author: Yanming Zhu

## Coverage decision

The existing S1-S14 set covers the linguistic trees, probe-coordinate QC,
time/cross-temporal decoding, baseline state, depth/artifact controls,
state-space/RSA, response archetypes, matched-noun unit census and heatmaps,
Kimi/neural position controls, alignment sensitivity, test sensitivity and
structural-position estimability.

Four evidence gaps remain and are promoted to new figures:

1. Recording yield and analysis inclusion by session.
2. Population-decoder specification and inference sensitivity.
3. Complete population-information diagnostics beyond the selected ablation panel.
4. Stimulus/acoustic and design-identifiability diagnostics.

## Figure contracts

### Recording yield and analysis inclusion

- Core conclusion: unit yield and quality composition vary across eight sessions;
  the primary artifact-clean good-unit decoder uses seven sessions and 456 good
  unit/run entries, with patients retained as the biological replicates.
- Archetype: quantitative grid.
- Panel map: session-level quality composition; overall inclusion funnel;
  session inclusion map.
- Reviewer risk: unit/run entries are nested and are not independent biological
  replicates.

### Population-decoder specification and inference sensitivity

- Core conclusion: the fixed top-five result lies at the upper end of a broad
  32-specification range, and held-sentence-pair support differs between pooled
  label permutation and exact five-patient inference.
- Archetype: asymmetric quantitative grid with a specification-curve hero panel.
- Panel map: 32 specifications; fixed decoder patient endpoints; held-pair
  patient endpoints and inferential calibration.
- Reviewer risk: specifications are correlated; the curve is a sensitivity
  analysis, not 32 independent replications.

### Population-information diagnostics

- Core conclusion: decoder performance depends disproportionately on a small
  train-ranked unit subset, but condition-preserving trial shuffling and exact
  patient-level cross-validated distance do not support an additional broad
  covariance-based population code.
- Archetype: quantitative grid.
- Panel map: ranked versus random ablation; simultaneous versus condition-wise
  shuffled activity; cross-validated population distance.
- Reviewer risk: computational ablation is not causal, and condition-wise
  shuffling is a covariance diagnostic rather than a prospective label null.

### Stimulus and model identifiability

- Core conclusion: active/passive class is perfectly coupled to marker family,
  noun slot is perfectly coupled to preceding-marker presence and serial time,
  and matched noun acoustics vary across positions; the design therefore does
  not isolate abstract voice or hierarchy.
- Archetype: quantitative grid.
- Panel map: marker-by-voice design matrix; serial-onset pairing; standardized
  acoustic differences; cross-validated unique voice contribution with and
  without word-identity control.
- Reviewer risk: stimulus items, not patients, are the units in acoustic panels;
  negative cross-validated R2 indicates lack of held-out predictive benefit and
  is not evidence of neuronal inhibition.

## Export contract

- Backend: Python/matplotlib only.
- Final size: 180 x 130 mm.
- Outputs: editable-text SVG/PDF, 600-dpi TIFF, PNG preview, source-data CSV and
  JSON metadata.
- Font: Arial/Helvetica/sans-serif, 6-8 pt at final size.
- Palette: neutral grey plus teal signal and gold/coral accents.
