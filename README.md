# Tan preprocessing, analysis and figure code

Author and maintainer: Yanming Zhu · Code license: MIT

**Code update: 2026-10-04.** The directory name retains the original 2026-08-09 snapshot date. This update incorporates the subsequent analysis and figure scripts used in `manuscript V3`, while preserving explicit access to historical analyses. No manuscript, figure or scientific result was regenerated as part of this code update.

## Release scope and result versions

This is a code archive, not a self-contained data deposit. Raw recordings, stimulus audio, clinical images, model weights and previously generated result tables must be supplied separately. The package contains the code needed to work with the documented project layout. Installation and static checks do not demonstrate a complete rerun from raw recordings.

Two distinctions are essential:

- **Three-condition verb analysis:** the analyzer now uses the corrected scrambled-verb anchors from 2026-08-12. Its default output is `neuropixel2025/data/combined_verb_three_condition_unit_activity_corrected_20260812/`; it does not overwrite the historical output directory. The saved corrected results contain 77 selected unit/run entries (31 active-, 18 passive-, 28 scrambled-preferring), whereas the unmodified manuscript and historical figures still cite 72 (30/18/24). These are existing result versions, not numbers recomputed during this update. Downstream scripts default to corrected input; historical input must be selected explicitly. Code updates alone do not reconcile the manuscript with the corrected result.
- **Fig. 2A/B:** the current comparison uses the same fixed L2 regularization `C=0.1` for the single-unit and top-five readouts. Nested regularization is a sensitivity analysis. The older sentence-pair comparison using `C=1` for the single unit remains historical/sensitivity code; it is not the producer of the updated main comparison. The manuscript still contains an older Methods paragraph with the unequal settings and was not edited here.

`d_A = sqrt(2) * Phi_inverse(AUC)` is an AUC-equivalent effect size. Predictive information is separately computed as held-out log-likelihood gain in bits/trial. Unit ranking, scaling and model fitting use training folds only; repeated out-of-fold probabilities are averaged per original trial before scoring. Patients are the independent biological units; runs and clusters are nested observations.

## Contents

- `scripts/`: preprocessing, analysis, plotting, input-version selection and package-validation modules.
- `by_stage/01_preprocessing/`, `02_analysis/`, `03_figures/`: navigation links to canonical scripts. Run the canonical script paths in the commands below.
- `by_stage/04_Supplementary_Figures_20260809/`: supplementary base-layout builders. This is their actual location inside the archive.
- `manuscript/revision_20260913/`, `revision_20260920/`, `revision_20260921_fig1hi/`, `revision_20260921_fig1i/`, `revision_20260921_s6s8/`: subsequent figure-generation code, preserving directory depth so relative paths remain correct. These directories contain code, not copies of the manuscript.
- `reports/neuron_venn_20260927/make_neuron_venn.py`: Fig. 3H overlap panel and source-table exporter.
- `analysis_contracts/`: five previously frozen exploratory geometry specifications.
- `tests/`: targeted decoding, anchor/version-selection and packaging checks.
- `SCRIPT_MANIFEST.tsv`: canonical code paths, source provenance and SHA-256 values. Source paths refer to the original project and need not exist inside this code-only archive.
- `requirements-analysis.txt`, `requirements-test.txt`, `requirements-figures.txt`, `package.json`: environment/dependency specifications; see limitations below.
- `VALIDATION.md`: the checks actually completed for this release and their limits.

Do not move individual scripts out of their directory hierarchy. Most derive the archive root from their own location; some preprocessing wrappers also require the current working directory to be the archive root.

## Environment and installation

The core analysis/test environment used for this update was Python 3.10.18 with the exact versions in `requirements-analysis.txt`. These pins describe the environment used for the focused checks; they do not establish the historical environment of every recording, Kilosort run or model extraction.

From an independent working copy of this archive:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python scripts/validate_code_package.py
python -m pytest -q tests
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. The commands below assume the archive root is the working directory.

Additional environments are task-specific:

| Task | Additional dependencies / requirements |
|---|---|
| Figure generation, Venn and PDF assembly | `python -m pip install -r requirements-figures.txt`; Arial for the exact Venn typography; a suitable Chinese font for Chinese-labelled figures |
| Supplementary PPTX base layout | Node.js 20.9 or newer; run `npm install` at the archive root to install `pptxgenjs`, `sharp`, `jszip` from `package.json` |
| Audio decoding | `ffmpeg` executable on `PATH` |
| Spike sorting | PyTorch and Kilosort, with the per-run original settings/hardware recorded separately; see below |
| Kimi representation extraction | The Apple-silicon MLX/MLX-VLM/transformers environment and the matching local model weights/tokenizer; not included in the core Python requirements |
| Cortical-surface figures | nibabel and nilearn; template data and localization inputs supplied separately |

The optional figure requirements include dependencies that were not all tested together in a fresh environment. The Venn geometry API is pinned to matplotlib-venn 0.11.10. See `VALIDATION.md` for the environments actually exercised. Model weights, fonts and third-party assets are not licensed by this archive's MIT license.

Sorting is not hardware-independent: the CPU branch in the retained Kilosort wrappers changes settings including `nblocks=0` and `templates_from_data=False`. Reproducing the original cluster inventory requires the actual input segment, Kilosort version, settings and hardware provenance for each run. The available local installation was Kilosort 4.1.7/PyTorch 2.2.2; that observation is not proof that all historical runs used those versions.

## Data layout and recording inventory

Stage the separately supplied data in a **working copy**, retaining these relative paths:

```text
archive_root/
  scripts/
  neuropixel2025/
    data/gz001/...                         # patient/run preprocessing products
    data/gz002/...                        
    data/gz003/...                        
    data/gz004/...                        
    data/gz005/...                        
    data/combined_*/...                   # analysis inputs and generated results
    files/实验设计/程序设计/neuropixel_m255_202511/
      ActPasSynSentence/scripts/stim_actpas/*.mp3
  models/...                              # only for model extraction
  outputs/...                             # event tables and unit screens
  reports/...                             # frozen tables and figure inputs
  manuscript/...                          # base PDF inputs and revision outputs
```

Do not replace an entire output tree with a writable link to the master project: many scripts write into `neuropixel2025/data`, `outputs`, `reports` or `manuscript`. Copy needed derivatives into a disposable working tree or mount read-only inputs at the required file paths. No data download URL or accession is asserted by this code-only archive; input access must accompany the separate data release.

| Participant | Recording run(s) |
|---|---|
| gz001 | `gz001t_g0_actpas_run1` |
| gz002 | `gz002_g0_actpas_run2` |
| gz003 | `gz003t2_g0_actpas_run1`, `gz003t2_g0_actpas_run4` |
| gz004 | `gz004t1_g0_actpas_run0`, `gz004t1_g3_actpas_run3` |
| gz005 | `gz005t0_g0_actpas_run0`, `gz005t0_g3_actpas_run3` |

The inclusive registry contains 1,782 run-specific Kilosort clusters across eight runs and five participants. The updated Fig. 2A/B comparison uses the seven eligible runs listed in `reports/nature_science_draft/supplementary/tables/supp_table_2_population_decoding.csv`, excluding `gz003t2_g0_actpas_run1`. Eligibility and unit sets are analysis-specific; do not substitute pooled clusters for independent participants or assume that every analysis uses eight runs.

## Analysis-to-code map and run order

The entries below distinguish primary/revised analyses from historical or sensitivity code. They are not a one-command end-to-end pipeline. First supply the required inputs, then run the relevant entry points in order.

| Component | Entry point / prerequisite |
|---|---|
| Run extraction and TTL/event synchronization | `extract_gz001_actpas.py` through `extract_gz005_actpas.py`; `update_gz004_behavior_events.py` |
| Sorting | `run_kilosort4_gz001_actpas.py` through `run_kilosort4_gz005_actpas.py` |
| Spike arrays, registry and artifact masks | `kilosort_spikes_to_firing_rate*.py`, `make_lenient_good_cluster_sets*.py`, `detect_ap_artifacts_1000hz*.py`, `summarize_artifact_overlap_with_trials*.py` |
| Acoustic onset correction | `estimate_actpas_function_word_audio_offsets*.py`; corrected verb anchor code described below |
| Function-word and noun screens | `analyze_active_passive_corrected_timepoint_screen_20260722.py`, `analyze_noun_slot_corrected_timepoint_screen_20260723.py` |
| Verb active/passive screen | `analyze_verb_active_passive_unit_activity.py`; the matched-verb panel tables used by Venn are produced by `make_fig3_verb_noun_slot_180mm.py` |
| Fig. 1H/I time-resolved / cross-time d_A | `analyze_active_passive_time_generalization.py`, followed by the September panel/assembly scripts below |
| Updated Fig. 2A/B | `analyze_unit_population_predictive_information.py`, followed by the September Fig. 2 builders |
| Historical decoder specifications and ablation inputs | `analyze_active_passive_network_encoding.py`, `analyze_unit_population_sentence_pair_transfer.py`, `analyze_population_encoding_candidate_panels.py`, `analyze_three_paper_borrowed_methods_20260724.py` |
| Corrected three-condition verb timing | `analyze_verb_three_condition_unit_activity.py`, `plot_verb_three_condition_population_timing.py`, timing/statistics downstream scripts below |
| Fig. 3H overlap | `reports/neuron_venn_20260927/make_neuron_venn.py` |
| Noun identity/position and Fig. 4 | `analyze_noun_role_structure_encoding.py`, `analyze_noun_semantic_vs_syntax_units.py`, `decode_noun_role_multifactor.py`, `make_fig4_noun_identity_position_kimi_20260806.py` |
| Kimi correspondence | `extract_kimi_vl_exploratory.py`, `extract_noun_firstchar_kimi.py`, `analyze_noun_firstchar_llm_mapping.py`, `plot_kimi_feature_triggered_neural_average.py`, `analyze_existing_neural_to_kimi_shift_mapping.py` |
| S6–S8 revised d_A panels | `manuscript/revision_20260921_s6s8/audit/build_supplementary_discriminability.py` |
| S10 timing sensitivity / S12 session composition | `analyze_active_passive_unsupervised_time_clusters_20260724.py`, `make_supp_fig11_temporal_band_validation_20260814.py` |
| Editable noun-panel exports | `export_noun_slot_figures_vector_20260815.py` |

Geometry/position controls retain their original scripts and contracts: `analyze_active_passive_geometry_transform_20260806.py`, `analyze_active_passive_manifold_geometry_20260806.py`, `analyze_active_passive_shared_latent_geometry_20260806.py`, `analyze_event_anchor_manifold_geometry_20260806.py`, `analyze_active_passive_nested_regularized_alignment_20260807.py`, `analyze_intro_driven_structural_transformation_20260807.py`. Their presence does not make all exploratory findings confirmatory.

### Updated Fig. 2 analysis and export

Required inputs include:

- `reports/nature_science_draft/supplementary/tables/supp_table_2_population_decoding.csv` (eligible run list).
- `outputs/active_passive_all_trial_sentences.csv` (matched sentence/trial metadata).
- `neuropixel2025/data/gz*/processed/ActPasSynSentence/event_related_active_passive/active_passive_unit_trial_response_metrics.csv` (baseline-corrected trial responses and quality labels).

```sh
python scripts/analyze_unit_population_predictive_information.py --cv-repeats 20 --fixed-c 0.1 --regularizations fixed,nested
python manuscript/revision_20260913/audit/build_discriminability_figures.py
python manuscript/revision_20260920/audit/build_descriptive_figures.py
```

The analyzer writes `neuropixel2025/data/combined_unit_population_predictive_information/` (`patient_values.csv`, trial/fold audit tables and metadata) and `reports/unit_population_predictive_information.md`. It also accepts `--output-dir` and `--report-path`; downstream figure builders expect the default tree unless their inputs are explicitly adapted.

The September 13 builder additionally reads the original decoder/specification and ablation products and writes derived tables under `manuscript/revision_20260913/source_data/`. The September 20 builder reads those tables plus frozen panels C/F/G from `reports/nature_science_draft/assembled_figures_20260805/source_data/fig2_population_180mm_revised/`, and writes `manuscript/revision_20260920/figures/` and `source_data_fig2/`. Copying only the last plotting script or supplying only `patient_values.csv` is insufficient.

### Fig. 1H/I revised panels

Inputs: `neuropixel2025/data/combined_active_passive_time_generalization/active_passive_time_generalization_matrices.npz`, its diagonal time-statistics CSV, and the earlier median-matrix export in `reports/nature_science_draft/assembled_figures_20260805/subplots/source_data/`. Full-figure assembly additionally requires the frozen base `manuscript/Fig1.pdf`.

```sh
python manuscript/revision_20260921_fig1i/audit/build_fig1i.py
python manuscript/revision_20260921_fig1i/audit/assemble_fig1.py
python manuscript/revision_20260921_fig1hi/audit/build_fig1h.py
python manuscript/revision_20260921_fig1hi/audit/assemble_fig1.py
```

The H panel is the median/IQR version corresponding to the caption; the alternative full-range display is not substituted. Outputs are within the corresponding revision directories. The assemblers do not edit or inject images into Word documents. Panel exporters can be used without the frozen base PDF; the two assembly commands cannot.

### Corrected three-condition verb analysis

The corrected analyzer requires the run spike arrays/trial bins, unit registries, stimulus audio, and active/passive verb-onset inputs at the paths used by its imported base scripts.

```sh
python scripts/analyze_verb_three_condition_unit_activity.py --unit-set usable_lenient
python scripts/plot_verb_three_condition_population_timing.py
python scripts/plot_verb_three_condition_timing_distribution.py
python scripts/plot_verb_three_condition_patient_stats.py
python scripts/summarize_verb_three_condition_population_differences.py
```

The analyzer always computes the corrected anchors. `TAN_VERB_THREE_CONDITION_OUT_DIR` and `TAN_VERB_THREE_CONDITION_REPORT_PATH` optionally redirect its output. Do not redirect corrected computation into the historical result directory.

Downstream commands support `--verb-version corrected_20260812` (default), `--verb-version legacy_20260723` (explicit historical tables), and `--verb-input-dir PATH`. Environment equivalents are `TAN_VERB_THREE_CONDITION_VERSION` and `TAN_VERB_THREE_CONDITION_OUT_DIR`. Selecting a historical table does not change the corrected analyzer into a historical analyzer. Custom directories require matching provenance.

`make_fig3_verb_noun_slot_180mm.py` also supports version selection for its three-condition timing panel. Its counts and timing comparisons are derived from the selected tables rather than assuming 72. Its figure output defaults to a version-labelled directory under `reports/nature_science_draft/` and can be redirected with `--output-dir` or `TAN_FIG3_OUT_DIR`. It remains a base composite; use the separate Venn exporter for the later Fig. 3H panel. Final manual layout changes in illustration software are not replayed by these scripts.

### Fig. 3H Venn

Required tables, joined by participant, run and unit ID:

```text
outputs/active_passive_corrected_timepoint_screen_20260722/corrected_unit_results.csv
outputs/noun_slot_corrected_timepoint_screen_20260723/corrected_noun_slot_unit_results.csv
outputs/fig3_verb_aligned_active_passive_20260805/verb_aligned_active_passive_unit_results.csv
```

```sh
python reports/neuron_venn_20260927/make_neuron_venn.py
```

The code checks a common 1,782-unit universe, group totals 172/279/148, union 425 and triple intersection 26 before export. Outputs include the panel, membership/count CSVs and QA JSON in the same report directory. The separate 719-unit three-condition analysis must not be substituted for the 148-unit binary verb set.

### Supplementary updates and noun exports

```sh
python manuscript/revision_20260921_s6s8/audit/build_supplementary_discriminability.py
python scripts/analyze_active_passive_unsupervised_time_clusters_20260724.py
python scripts/make_supp_fig11_temporal_band_validation_20260814.py
python scripts/export_noun_slot_figures_vector_20260815.py
```

S6–S8 require the specification tables in `reports/three_paper_borrowed_analyses_20260724/tables`, frozen Fig. 2 source tables, and `combined_active_passive_population_candidate_panels`, `combined_active_passive_baseline_state`, `combined_active_passive_sentence_pair_transfer` under `neuropixel2025/data/`.

The GMM/LOPO sensitivity script reads response profiles in `reports/active_passive_go_style_interpretable_20260723/` and writes `reports/active_passive_unsupervised_time_clustering_20260724/`. The following exporter reads those outputs and the archetype session-composition table. Its historical filename still says Fig11; the current manuscript uses this evidence in S10 and S12. It validates descriptive time bands; it does not establish natural neuronal types or replace the fixed-band main-figure grouping. The earlier README's blanket exclusion of this upstream script no longer applies.

The noun exporter reads the frozen `outputs/noun_slot_corrected_timepoint_screen_20260723/` tables and matrices, then writes `reports/noun_slot_unit_analysis_20260723/vector_exports_20260815/`; it does not rerun the unit screen.

### Historical supplementary base layout

The builders under `by_stage/04_Supplementary_Figures_20260809/` retain the historical 24-figure layout and Table S1, not an automatically renumbered final supplementary deck. The previous `supplementary_figure_manifest.json` and gap-audit document are historical records of the earlier layout, not this release's code-validation report. Later S6–S8, S10/S12 and consolidated patient-depth replacements are produced separately.

```sh
python by_stage/04_Supplementary_Figures_20260809/build_missing_supplementary_figures.py
python by_stage/04_Supplementary_Figures_20260809/build_patient_depth_supplement.py
python by_stage/04_Supplementary_Figures_20260809/build_combined_patient_depth_figure.py
node by_stage/04_Supplementary_Figures_20260809/build_supplementary_figures.mjs --help
```

The Node builder accepts `--assets-dir`, `--source-data-dir`, `--figures-dir`, `--input-root`, `--output-dir` and the documented `TAN_SUPPLEMENTARY_*` environment variables. Static assets must include `ba_tree.png`, `bei_tree.png`, the matched-sentence raster, and the five patient-depth PNGs; these images are not included in this code update. Patient-depth builders require the five `*_unit_depth_activation_metrics.csv` tables. Other report images are enumerated in the Node builder's `paths` object. Table S1 values are frozen historical records embedded in the builder; its CSV path records provenance, not a live import that refreshes those values.

The Node output is explicitly named `Tan_Supplementary_Base_Layout_20260810.pptx`, with `supplementary_base_layout_manifest.json`. It should not be mistaken for the final manuscript supplement without applying and checking the later panel substitutions.

## Packaging validation and retired entries

```sh
python scripts/validate_code_package.py --output /tmp/tan_code_validation.json
python -m pytest -q tests
```

The validator checks manifest coverage/hashes, symlinks, Python syntax, available Node syntax checking and package-local imports without importing analysis modules. Missing third-party dependencies are reported as skips, not interpreted as available. It does not check data access or recompute scientific results.

The unused historical M1/R0 entry point `make_submission_figure_variants.py` was removed from this archive and its navigation/manifest entries because it depended on the excluded `condition_contrast_figures` package and is not an entry point for the current Fig. 1–4. The older 50-ms screen, superseded `make_fig2_population_180mm.py`, and alternative condition-contrast pipeline remain outside this package. Historical scripts retained for sensitivity inputs are identified in the table above.

## Interpretation and license

The active/passive contrast remains confounded with construction-marker identity and acoustics. Noun position covaries with serial time, preceding-marker context, acoustics and baseline state. Significant response screens, held-out decoding and exploratory geometry are different evidence types. Do not interpret run-specific cluster entries as independently sampled or longitudinally identified neurons.

The source code and original documentation are licensed under MIT; see `LICENSE` and `AUTHORS.md`. This license does not grant rights to patient data, clinical records, stimulus recordings, model weights, third-party software, fonts or visual assets. Those materials retain their own access and licensing terms.
