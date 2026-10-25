import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import fsp from "node:fs/promises";
import { parseArgs } from "node:util";

/**
 * Historical 24-figure supplementary base layout (August 2026), plus Table S1.
 * This is not the final renumbered supplementary deck. Later quantitative
 * replacements are built by manuscript/revision_20260921_s6s8/audit/
 * build_supplementary_discriminability.py and scripts/
 * make_supp_fig11_temporal_band_validation_20260814.py; the patient-depth
 * consolidation is built by build_combined_patient_depth_figure.py.
 * Consult the package README for current figure mapping. Assets and tables
 * are external inputs; running this builder never updates source manuscript files.
 */
const { values } = parseArgs({ options: {
  "assets-dir": { type: "string" },
  "source-data-dir": { type: "string" },
  "figures-dir": { type: "string" },
  "input-root": { type: "string" },
  "output-dir": { type: "string" },
  "help": { type: "boolean", short: "h" },
} });
if (values.help) {
  console.log(`Build the historical 24-figure supplementary base layout, plus Table S1.
Usage: node build_supplementary_figures.mjs [options]
  --assets-dir DIR       Static assets directory (TAN_SUPPLEMENTARY_ASSETS_DIR).
  --source-data-dir DIR  Source table directory (TAN_SUPPLEMENTARY_SOURCE_DATA_DIR).
  --figures-dir DIR      Base panels from build_missing_supplementary_figures.py
                        (TAN_SUPPLEMENTARY_FIGURES_DIR).
  --input-root DIR       Data root containing reports/ (TAN_SUPPLEMENTARY_INPUT_ROOT).
  --output-dir DIR       Output directory; defaults to this script's directory.
Install Node dependencies with: npm install pptxgenjs sharp jszip
This builder preserves historical figure numbers; see README for later S6-S8,
S10/S12 and consolidated patient-depth replacements.`);
  process.exit(0);
}
const runtimeRequire = createRequire(import.meta.url);
const pptxgen = runtimeRequire("pptxgenjs");
const sharp = runtimeRequire("sharp");
const JSZip = runtimeRequire("jszip");

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(values["input-root"] || process.env.TAN_SUPPLEMENTARY_INPUT_ROOT || path.join(HERE, "..", ".."));
const ASSETS = path.resolve(values["assets-dir"] || process.env.TAN_SUPPLEMENTARY_ASSETS_DIR || path.join(HERE, "assets"));
const SOURCE_DATA = path.resolve(values["source-data-dir"] || process.env.TAN_SUPPLEMENTARY_SOURCE_DATA_DIR || path.join(HERE, "source_data"));
const FIGURES = path.resolve(values["figures-dir"] || process.env.TAN_SUPPLEMENTARY_FIGURES_DIR || path.join(HERE, "new_figures"));
const OUTPUT_DIR = path.resolve(values["output-dir"] || HERE);
const OUT = path.join(OUTPUT_DIR, "Tan_Supplementary_Base_Layout_20260810.pptx");
const MANIFEST = path.join(OUTPUT_DIR, "supplementary_base_layout_manifest.json");
const TABLE_S1_SOURCE = path.join(SOURCE_DATA, "Supplementary_Table_S1_recording_summary.csv");
await fsp.mkdir(OUTPUT_DIR, { recursive: true });

const pptx = new pptxgen();
pptx.defineLayout({ name: "WIDE", width: 13.333, height: 7.5 });
pptx.layout = "WIDE";
pptx.author = "Yanming Zhu";
pptx.company = "ShanghaiTech University";
pptx.subject = "Supplementary figures for the Tan Neuropixels manuscript";
pptx.title = "Supplementary Figures - Single-neuronal encoding of syntactic computations";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  lang: "en-US",
};
pptx.margin = 0;

const C = {
  ink: "16312C",
  deep: "102A26",
  teal: "147D70",
  tealPale: "DDEEEA",
  coral: "D9644A",
  coralPale: "F8E2DC",
  gold: "D9A01C",
  goldPale: "F6EBCB",
  cream: "F5F4EF",
  white: "FFFFFF",
  gray: "64706C",
  gray2: "8B9692",
  line: "D5DEDA",
  legend: "E8F0ED",
  blue: "2572B7",
};

const FONT = "Arial";
const FIG_BOX = { x: 0.53, y: 0.80, w: 8.18, h: 6.08 };
const LEGEND_BOX = { x: 9.17, y: 0.68, w: 3.73, h: 6.42 };

const paths = {
  baTree: path.join(ASSETS, "ba_tree.png"),
  beiTree: path.join(ASSETS, "bei_tree.png"),
  recordingYield: path.join(FIGURES, "S2_recording_yield_and_inclusion.png"),
  recordingExample: path.join(ASSETS, "gz002_run2_matched_sentence_raster_example.png"),
  probeBoundary: path.join(ROOT, "reports", "probe_tissue_boundary_candidates_20260730", "figures", "Fig_probe_boundary_candidates_all_patients.png"),
  patientDepthImages: ["gz001", "gz002", "gz003", "gz004", "gz005"].map((subject) =>
    path.join(ASSETS, "patient_depth_activation", `${subject}_depth_activation_left_map.png`),
  ),
  patientDepthSourceData: ["gz001", "gz002", "gz003", "gz004", "gz005"].map((subject) =>
    path.join(SOURCE_DATA, "patient_depth_activation", `${subject}_unit_depth_activation_metrics.csv`),
  ),
  timeGeneralization: path.join(ROOT, "reports", "nature_science_draft", "supplementary", "figures", "supp_fig1_time_generalization.png"),
  decoderSensitivity: path.join(FIGURES, "S5_decoder_specification_and_inference.png"),
  populationDiagnostics: path.join(FIGURES, "S6_population_information_diagnostics.png"),
  baselineState: path.join(ROOT, "reports", "nature_science_draft", "supplementary", "figures", "supp_fig4_baseline_state.png"),
  depthArtifact: path.join(ROOT, "reports", "nature_science_draft", "supplementary", "figures", "supp_fig5_depth_artifact_controls.png"),
  stateSpace: path.join(ROOT, "reports", "nature_science_draft", "supplementary", "figures", "supp_fig3_state_space_rsa.png"),
  clusterComposition: path.join(ROOT, "reports", "active_passive_go_style_interpretable_20260723", "Fig_GO_style_interpretable_cluster_composition.png"),
  nounScreen: path.join(ROOT, "reports", "noun_slot_unit_analysis_20260723", "figures", "Fig_corrected_noun_slot_screen_summary.png"),
  nounClassControl: path.join(ROOT, "reports", "noun_slot_unit_analysis_20260723", "figures", "Fig_corrected_noun_slot_noun_class_control.png"),
  subjectHeatmap: path.join(ROOT, "reports", "noun_slot_unit_analysis_20260723", "figures", "Fig_corrected_noun_slot_subject_gt_predicate_stacked_heatmap.png"),
  postMarkerHeatmap: path.join(ROOT, "reports", "noun_slot_unit_analysis_20260723", "figures", "Fig_corrected_noun_slot_subject_lt_predicate_stacked_heatmap.png"),
  stimulusIdentifiability: path.join(FIGURES, "S14_stimulus_acoustic_identifiability.png"),
  positionControls: path.join(ROOT, "reports", "fig4_extended_data_20260807", "figures", "ExtendedDataFig4_position_controls.png"),
  alignmentSpecs: path.join(ROOT, "reports", "active_passive_nested_regularized_alignment_20260807", "figures", "Fig_6_combined_unit_specification_comparison.png"),
  testSensitivity: path.join(ROOT, "reports", "active_passive_nested_regularized_alignment_20260807", "figures", "Fig_7_good_only_statistical_test_sensitivity.png"),
  positionValidation: path.join(ROOT, "reports", "intro_driven_structural_transformation_20260807", "figures", "Extended_Data_Fig_9_validation_sensitivity.png"),
};

const patientDepthPanels = [
  { subject: "gz001", entries: 201, runs: 1, nominal: 22 },
  { subject: "gz002", entries: 352, runs: 1, nominal: 6 },
  { subject: "gz003", entries: 366, runs: 2, nominal: 13 },
  { subject: "gz004", entries: 440, runs: 2, nominal: 27 },
  { subject: "gz005", entries: 423, runs: 2, nominal: 14 },
];

const legends = [
  {
    title: "Syntactic analyses of matched Mandarin BA and BEI sentences",
    body: "a, One analysis of the active BA construction, in which chongzi (insects) occupies the sentence-initial subject/Agent position and shuye (leaves) occurs after BA as the Patient noun within the predicate. b, One analysis of the matched passive BEI construction, in which shuye occupies the sentence-initial subject/Patient position and chongzi occurs after BEI as the demoted Agent; ti marks the lower position associated with the promoted Patient under the illustrated movement analysis. The lexical items and thematic roles are retained while sentence-initial and post-marker positions are exchanged. These trees provide a linguistic interpretation of the stimulus manipulation; the neural analyses use the empirically neutral labels sentence-initial noun and post-marker noun and do not adjudicate among competing formal syntactic analyses.",
    sources: [paths.baTree, paths.beiTree],
  },
  {
    title: "Recording yield and analysis inclusion across sessions",
    body: "a, Session-resolved Kilosort quality composition for all 1,782 registered unit/run entries: 461 were labelled good and 1,321 multi-unit activity (MUA). b, Inclusion funnel from the registered set to good entries and the seven-session primary population decoder. c, Session map and good-entry counts. gz003 run1 contributed five good entries to the recording census but was excluded from the fixed decoder because only seven artifact-clean trials (two active and five passive) remained, which was insufficient for the fixed cross-validation scheme. The decoder therefore used 456 good unit/run entries across seven sessions. Unit/run entries and sessions are nested observations; the independent biological replicates are the five patients.",
    sources: [paths.recordingYield],
  },
  {
    title: "Example spectrograms and depth-sorted spiking for a matched sentence pair",
    body: "Example recording from gz002 run2. Left, the active sentence 'Ying'er BA niunai he-wan-le' (The baby finished drinking the milk in the BA construction). Right, the matched passive sentence 'Niunai BEI ying'er he-wan-le' (The milk was finished by the baby in the BEI construction). Spectrograms are shown above the simultaneously recorded spike rasters. Each grey point is a spike from one of 352 Kilosort clusters, plotted at that cluster's median probe depth. Solid red lines mark detected speech onset and sound-file end; dashed red lines mark construction-marker onset. Blue and coral right-edge triangles mark nominal active-greater-than-passive and passive-greater-than-active candidate clusters, respectively, from an uncorrected P < 0.05 screen. This single pair illustrates population sampling, probe-depth coverage and heterogeneous temporal activity; it is not patient-level evidence, the triangles do not denote corrected significance, and probe depth is not an anatomical cortical-layer assignment.",
    sources: [paths.recordingExample],
  },
  {
    title: "Patient-resolved probe-coordinate boundary quality control",
    body: "The black traces show the multi-feature physiological change score along the normalized probe y-coordinate for eight recording runs. Green circles denote boundary candidates that met the fixed bootstrap-support, feature-agreement, level-contrast, confidence-interval-width and edge-exclusion criteria; the gold diamond denotes a tentative candidate, and grey crosses denote unsupported local transitions. Five of eight runs contained at least one supported transition, but only gz003 run4 contained two supported transitions. The near-tip transition in the two gz003 runs was reproducible in probe coordinates. Without an independently registered trajectory or histology, these transitions cannot be assigned to pia, white matter or cortical layers; all depth analyses therefore retain an anatomically unassigned, relative-probe status.",
    sources: [paths.probeBoundary],
  },
  ...patientDepthPanels.map((panel, idx) => ({
    title: `Unit/run response differences across probe depth in ${panel.subject}`,
    body: `All ${panel.entries} Kilosort unit/run entries from ${panel.runs} recording ${panel.runs === 1 ? "run" : "runs"} in patient ${panel.subject} are plotted by median estimated probe depth against the absolute active-passive difference in baseline-corrected firing rate; the upper histogram shows the response-difference distribution. Grey points did not meet nominal P < 0.05. Blue and coral points denote nominal active-greater-than-passive and passive-greater-than-active candidates, respectively, from two-sided Mann-Whitney U tests; black edges denote the Kilosort good label. Labels identify candidate units; for multi-run patients, r#·u# denotes the recording-run number and Kilosort unit ID. Point size is display-scaled by spike count. Activation is the mean 0-0.8-s response minus the -1-to-0-s baseline. ${panel.nominal} entries met uncorrected P < 0.05 and none survived within-run BH-FDR at q < 0.05. Entries and runs are nested within the patient; probe depth is an unregistered recording coordinate, not anatomical cortical layer, and active/passive class remains confounded with marker family and acoustics.`,
    sources: [paths.patientDepthImages[idx], paths.patientDepthSourceData[idx]],
  })),
  {
    title: "Time-resolved and cross-temporal construction decoding by run",
    body: "a, Diagonal decoding with a fixed training-fold-selected top-five-unit L2 logistic model in 200-ms windows advanced in 50-ms steps. Coloured lines denote seven artifact-clean, good-unit runs and the black line denotes the run median. b, Median train-time by test-time AUC matrix. c, The seven run-specific matrices. Across 0-800 ms after construction-marker onset, the median of run mean diagonal AUCs was 0.531 and the median mean cross-temporal AUC was 0.510. The descriptive median diagonal maximum was AUC = 0.67 at 0.30 s, but no time bin survived BH-FDR correction. The predominantly local and run-variable generalization pattern supports a transient construction-associated state rather than a stable time-generalized code. Active/passive construction remains collinear with marker family and marker acoustics.",
    sources: [paths.timeGeneralization],
  },
  {
    title: "Population-decoder specification and inference sensitivity",
    body: "a, Patient-median AUC and bootstrap 95% confidence interval for 32 correlated analysis specifications spanning artifact handling, unit set and decoder family, ordered by their median. The specifications ranged from 0.46 to 0.69; 27 of 32 medians exceeded 0.5. The gold star marks the prespecified artifact-clean, good-unit, training-fold-selected fixed top-five L2 model (median AUC = 0.69; exact one-sided patient sign P = 0.1875). b, Patient endpoints for the nested single-unit and fixed top-five primary decoders. c, Leave-one-sentence-pair-out endpoints. The held-pair top-five median was 0.66, with pooled paired-label permutation P = 0.024 but exact five-patient sign P = 0.1875. The curve is a sensitivity analysis, not 32 independent replications; patient-level inference remains primary.",
    sources: [paths.decoderSensitivity],
  },
  {
    title: "Population-information diagnostics",
    body: "a, Patient-median decoder AUC after removing increasing numbers of training-fold-ranked or randomly selected units; shading denotes the patient interquartile range. Ranked removal reduced AUC more than random removal at one, two and five removed units (one-sided exact P = 0.03125), indicating disproportionate readout dependence on a small train-ranked subset. b, Simultaneous population AUC compared with condition-preserving independent trial shuffling across units. For the top-five model, the patient-median difference was 0.005 (two-sided exact P = 1.0). c, Cross-validated active-passive population distance for all units and the top five; the top-five median was 0.148 (one-sided exact P = 0.0625). Ablation is computational rather than causal, and the condition-preserving shuffle diagnoses covariance beyond condition means rather than providing a prospective label null.",
    sources: [paths.populationDiagnostics],
  },
  {
    title: "Baseline-state controls for construction decoding",
    body: "a, Leave-one-sentence-pair-out AUC from the pre-marker baseline, raw evoked response, baseline-corrected response and response residualized against unitwise baseline. Across seven artifact-clean good-unit runs, median AUCs were 0.45, 0.53, 0.66 and 0.40, respectively. b, Fraction of held-out pairs projected in the active-greater-than-passive direction. c, Cosine similarity between axes estimated from baseline, evoked and residualized responses. d, Distribution of unitwise trial-level baseline-response correlations. The baseline is the -1 to 0 s pre-marker context, not a silent pre-sentence resting period. Baseline decoding was not above chance, and baseline and evoked axes were not positively aligned. These are exploratory run-level diagnostics; runs are nested within five patients.",
    sources: [paths.baselineState],
  },
  {
    title: "Relative-depth and IED/artifact controls",
    body: "a,b, Signed and absolute construction effects plotted against normalized probe depth separately for each patient. c, Patient summaries across relative-depth tertiles. These tertiles are pseudo-laminar bins and are not anatomical cortical layers. d, Fraction of trial windows flagged for artifact in each condition and run. e, Decoding from artifact/spike-field load alone across progressively stricter artifact thresholds. f, Raw versus artifact-controlled unit effects. No consistent patient-level depth gradient was observed; the gz005 absolute-effect gradient was an individual-patient observation. Artifact burden was condition-imbalanced in several runs, but the main population advantage remained visible across all-trial/artifact-clean and all-cluster/good-unit specifications (main Fig. 2d). These panels are diagnostic and do not establish laminar organization or remove the marker-acoustic confound.",
    sources: [paths.depthArtifact],
  },
  {
    title: "Population state-space and representational-geometry diagnostics",
    body: "a, Principal-component projections of artifact-clean, good-unit marker-aligned responses for each run; lines link matched sentence pairs. b, Demixed-style marginal variance fractions. Sentence-pair structure (median 0.464) and word identity (median 0.256) dominated the construction-associated component (median 0.059). c, Spearman correlations between neural representational dissimilarity matrices and construction, sentence-pair and word-identity models. d, Corresponding multiple-regression RSA coefficients. Global RSA correlations were near zero for construction. The overall pattern is consistent with heterogeneous population geometry rather than broad categorical separation of active and passive trials. Run-level summaries are descriptive, and marker identity remains collinear with construction.",
    sources: [paths.stateSpace],
  },
  {
    title: "Cross-session composition across six possible response archetypes",
    body: "Each of the 172 construction-modulated unit/run entries was assigned by two descriptive features: the direction of its qualifying active-minus-passive response and the latency tertile of its largest absolute qualifying difference (early, middle or late). Bars show the fraction assigned to the six possible archetypes within each recording session; labels at right give the number of qualifying entries. Both response directions and multiple latency groups occurred across patients and sessions, although their proportions varied and one session contained no entries after the corrected screen. The archetypes summarize response phenotypes and are neither anatomical cell classes nor causal network modules. Construction direction is inseparable from marker family in the present stimuli.",
    sources: [paths.clusterComposition],
  },
  {
    title: "Census and cross-noun-class replication of matched-noun slot effects",
    body: "a, Patient- and session-resolved counts for the 279 of 1,782 unit/run entries passing the matched-noun, baseline-corrected time-course screen. Of these, 163 showed greater activity for the sentence-initial noun, 115 showed greater activity for the post-function-word noun and one showed mixed directions. b, Replication after dividing the 20 nouns into Agent and Patient strata: 33 entries reproduced in the same direction in both strata, 18 reversed direction, 109 passed in only one stratum, 118 passed only in the pooled analysis and one was mixed. Same-direction replication occurred in four of five patients. The screen used P < 0.025 for at least six consecutive 1-ms bins in 0-0.8 s; entries are nested within patients, and lower stratum sample size reduces power.",
    sources: [paths.nounScreen, paths.nounClassControl],
  },
  {
    title: "Matched-noun units preferring the sentence-initial slot",
    body: "Rows are baseline-corrected unit/run entries that showed higher activity when the same noun occurred sentence-initially than after a construction marker. The left and middle heatmaps show firing rates for the sentence-initial and post-function-word occurrences; the right heatmap shows their difference. Rows are grouped by patient, aligned to the corrected acoustic onset of the noun and displayed from -1 to 2 s. The statistical screen used the 0-0.8-s interval with P < 0.025 for at least six consecutive 1-ms bins. Display-only stripe quality control retained 139 of 163 screened entries; it did not redefine the inferential set. These heatmaps show heterogeneity in response timing and background activity and should not be read as evidence for a homogeneous subject-selective cell class.",
    sources: [paths.subjectHeatmap],
  },
  {
    title: "Matched-noun units preferring the post-function-word slot",
    body: "Rows are baseline-corrected unit/run entries that showed higher activity when the same noun occurred after a construction marker than sentence-initially. The left and middle heatmaps show sentence-initial and post-function-word firing rates; the right heatmap shows sentence-initial minus post-function-word activity. Rows are grouped by patient and aligned to corrected noun onset. The matched-noun screen used P < 0.025 for at least six consecutive 1-ms bins in 0-0.8 s. Display-only stripe quality control retained 99 of 115 screened entries; it did not alter the primary census. The distributed timing and the concentration of entries in particular sessions reinforce that unit counts are descriptive and nested, not independent patient-level evidence.",
    sources: [paths.postMarkerHeatmap],
  },
  {
    title: "Stimulus acoustics and design identifiability",
    body: "a, Sentence counts for the six construction markers by active/passive class. BA, GEI and JIANG occurred only in active sentences, whereas BEI, RANG and JIAO occurred only in passive sentences (20 sentences; Cramer's V = 1.00). b, Sentence-relative onset of the 20 matched nouns when sentence-initial or post-marker. All initial nouns began at sentence onset and lacked a preceding marker; all post-marker nouns occurred later and followed a marker. c, Post-marker-minus-initial differences in six standardized acoustic features for the same 20 nouns. d, Run-median unique cross-validated R2 for the neural active/passive condition candidate before and after adding construction-marker identity; seven runs are nested in five patients. Because active/passive class and marker family are collinear, the residual is an identifiability diagnostic, not a clean abstract-voice estimate. Negative cross-validated R2 denotes lack of held-out predictive benefit, not neuronal inhibition.",
    sources: [paths.stimulusIdentifiability],
  },
  {
    title: "Kimi layer, input-perturbation and neural controls for noun identity and position",
    body: "a, Held-out Kimi decoding across the embedding and 27 hidden-state levels. Noun identity generalized across positions at the embedding and first hidden level but approached its 5% chance level in deeper states, whereas position and cross-role position approached ceiling. b, Cross-role position decoding after noun substitution, target masking, marker neutralization and target-only input. These perturbations preserve different portions of the prefix and are descriptive. c, Cross-role neural position decoding from pre-noun baseline, raw and baseline-corrected responses for five patients. d, Patient-equal time-resolved decoding; no window survived within-target Bonferroni correction. e, Permutation calibration of cross-position noun-identity decoding (observed 4.75%; P = 0.615). The shared neural-Kimi component is therefore dominated by position/local prefix context rather than stable noun-specific alignment.",
    sources: [paths.positionControls],
  },
  {
    title: "Nested rank-4 alignment across estimators and unit specifications",
    body: "Target-noun responses averaged from 0 to 800 ms were evaluated with identical leave-two-sentence-pairs-out outer folds. Original Procrustes, noise-whitened Procrustes and regularized partial least squares were compared in good-only and usable-lenient unit sets; noise-covariance and spectral shrinkage parameters were selected only within inner training folds. Points denote patients, grey lines connect estimates from the same patient and black bars denote patient means. Numbers are one-sided exhaustive patient max-statistic family-wise-error P values across the three estimators within each unit specification. Panel labels are rounded to three decimals; the exact minimum is 0.0625. The candidate correspondence did not become stronger under more complex estimators and was smaller under the lenient unit set. This post-result exploratory analysis does not isolate construction from marker, acoustic, serial-position or sentence-form differences.",
    sources: [paths.alignmentSpecs],
  },
  {
    title: "Statistical-test sensitivity of the good-only rank-4 Procrustes result",
    body: "a, Held-sentence-pair retrieval margins for the five patients; the mean was 0.0866 and four of five effects were positive. b, One-sided P values from distribution-free and normal-model-dependent tests applied to the same five patient effects. c, The exact 32-value sign-flip null distribution; two sign patterns were at least as large as the observed mean, giving P = 0.0625. d, The continuous t(4) reference distribution, which yields P = 0.0177 but assumes normally distributed patient effects; Holm correction across three alignment methods gives P = 0.0498. The distribution-free analyses do not cross 0.05, so the parametric result is an assumption-sensitive secondary observation. Patients, not runs, folds or neurons, are the independent inferential units.",
    sources: [paths.testSensitivity],
  },
  {
    title: "Validation and estimability limits of the structural-position analysis",
    body: "a, Fraction of the position vector remaining uniquely estimable after adding acoustic, serial-time and structural covariates. b, Variance-inflation audit: serial time is strongly collinear with position, and the theory-specific structural term is effectively non-estimable. c, Parallelism score and cross-condition generalization above chance for good-only and usable-lenient units. d, Participation ratio relative to the number of units. e, Exact patient-level time-point P values; with five patients the minimum attainable one-sided sign-flip P is 0.03125, above the 0.0125 fixed-window threshold. f, Sentence-initial nouns occur near sentence onset, whereas post-marker nouns necessarily follow a marker. The analysis supports a distributed position/construction-related state, but cannot separate hierarchy from serial time, preceding-marker presence and local context.",
    sources: [paths.positionValidation],
  },
];

function shadow() {
  return { type: "outer", color: C.ink, opacity: 0.10, blur: 2, angle: 45, offset: 1 };
}

function addBase(slide, index, legend) {
  slide.background = { color: C.cream };
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.44, y: 0.18, w: 2.56, h: 0.36, rectRadius: 0.06,
    fill: { color: C.teal }, line: { color: C.teal },
  });
  slide.addText(`SUPPLEMENTARY FIGURE S${index}`, {
    x: 0.56, y: 0.245, w: 2.32, h: 0.20, fontFace: FONT, fontSize: 9.3,
    bold: true, color: C.white, align: "center", margin: 0, charSpacing: 0.8,
  });
  slide.addText("Single-neuronal encoding of syntactic computations", {
    x: 6.04, y: 0.255, w: 6.84, h: 0.18, fontFace: FONT, fontSize: 8.5,
    color: C.gray, align: "right", margin: 0,
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.42, y: 0.68, w: 8.45, h: 6.42, rectRadius: 0.05,
    fill: { color: C.white }, line: { color: C.line, width: 0.8 }, shadow: shadow(),
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    ...LEGEND_BOX, rectRadius: 0.05,
    fill: { color: C.legend }, line: { color: C.line, width: 0.8 },
  });
  slide.addText("FIGURE LEGEND", {
    x: 9.43, y: 0.95, w: 3.18, h: 0.25, fontFace: FONT, fontSize: 9.2,
    bold: true, color: C.teal, margin: 0, charSpacing: 0.8,
  });
  slide.addText(legend.title, {
    x: 9.43, y: 1.28, w: 3.18, h: 0.78, fontFace: FONT, fontSize: 13.2,
    bold: true, color: C.ink, margin: 0, breakLine: false, fit: "shrink", valign: "top",
  });
  slide.addText(legend.body, {
    x: 9.43, y: 2.17, w: 3.18, h: 4.68, fontFace: FONT, fontSize: 10.0,
    color: C.deep, margin: 0, breakLine: false, fit: "shrink", valign: "top",
    paraSpaceAfterPt: 4, lineSpacingMultiple: 0.95,
  });
}

async function imageMetadata(imagePath) {
  const metadata = await sharp(imagePath).metadata();
  if (!metadata.width || !metadata.height) throw new Error(`Cannot read image dimensions: ${imagePath}`);
  return { width: metadata.width, height: metadata.height };
}

async function addContainedImage(slide, imagePath, box, altText = "Supplementary figure") {
  const meta = await imageMetadata(imagePath);
  const scale = Math.min(box.w / meta.width, box.h / meta.height);
  const w = meta.width * scale;
  const h = meta.height * scale;
  const x = box.x + (box.w - w) / 2;
  const y = box.y + (box.h - h) / 2;
  slide.addImage({ path: imagePath, x, y, w, h, altText });
  return { x, y, w, h };
}

function addReplacementFigureTitle(slide, pos, title) {
  slide.addShape(pptx.ShapeType.rect, {
    x: pos.x, y: pos.y, w: pos.w, h: 0.46,
    fill: { color: C.white }, line: { color: C.white },
  });
  slide.addText(title, {
    x: pos.x + 0.18, y: pos.y + 0.06, w: pos.w - 0.36, h: 0.28,
    fontFace: FONT, fontSize: 15.5, bold: true, color: C.ink,
    align: "center", margin: 0, fit: "shrink",
  });
}

async function addSingleImageSlide(index, imagePath, options = {}) {
  const slide = pptx.addSlide();
  const legend = legends[index - 1];
  addBase(slide, index, legend);
  const pos = await addContainedImage(slide, imagePath, options.box ?? FIG_BOX, legend.title);
  if (options.replaceEmbeddedTitle) addReplacementFigureTitle(slide, pos, options.replaceEmbeddedTitle);
  return slide;
}

// S1 - trees and role/position mapping.
{
  const index = 1;
  const slide = pptx.addSlide();
  addBase(slide, index, legends[index - 1]);
  slide.addText("Matched lexical items retain semantic roles while exchanging surface positions", {
    x: 0.82, y: 0.94, w: 7.76, h: 0.38, fontFace: FONT, fontSize: 16.5,
    bold: true, color: C.ink, align: "center", margin: 0, fit: "shrink",
  });
  slide.addText("a  BA construction", {
    x: 0.78, y: 1.42, w: 3.62, h: 0.28, fontFace: FONT, fontSize: 12.2,
    bold: true, color: C.teal, margin: 0,
  });
  slide.addText("b  BEI construction", {
    x: 4.67, y: 1.42, w: 3.62, h: 0.28, fontFace: FONT, fontSize: 12.2,
    bold: true, color: C.coral, margin: 0,
  });
  await addContainedImage(slide, paths.baTree, { x: 0.72, y: 1.76, w: 3.82, h: 3.95 }, "BA syntactic tree");
  await addContainedImage(slide, paths.beiTree, { x: 4.62, y: 1.76, w: 3.82, h: 3.95 }, "BEI syntactic tree");
  const tags = [
    { x: 1.03, w: 2.08, text: "same nouns", fill: C.tealPale, color: C.teal },
    { x: 3.26, w: 2.20, text: "same semantic roles", fill: C.goldPale, color: "8A6515" },
    { x: 5.61, w: 2.45, text: "positions exchanged", fill: C.coralPale, color: C.coral },
  ];
  for (const tag of tags) {
    slide.addShape(pptx.ShapeType.roundRect, {
      x: tag.x, y: 6.22, w: tag.w, h: 0.40, rectRadius: 0.06,
      fill: { color: tag.fill }, line: { color: tag.fill },
    });
    slide.addText(tag.text, {
      x: tag.x, y: 6.30, w: tag.w, h: 0.20, fontFace: FONT, fontSize: 10,
      bold: true, color: tag.color, align: "center", margin: 0,
    });
  }
}

await addSingleImageSlide(2, paths.recordingYield);
{
  const index = 3;
  const slide = pptx.addSlide();
  addBase(slide, index, legends[index - 1]);
  slide.addText("Matched-sentence spectrograms and depth-sorted population spiking", {
    x: 0.82, y: 0.94, w: 7.76, h: 0.34, fontFace: FONT, fontSize: 16.0,
    bold: true, color: C.ink, align: "center", margin: 0, fit: "shrink",
  });
  await addContainedImage(
    slide,
    paths.recordingExample,
    { x: 0.66, y: 1.34, w: 8.00, h: 3.82 },
    "Matched active and passive sentence spectrograms with depth-sorted spikes",
  );
  const tags = [
    { x: 0.98, w: 2.02, text: "352 Kilosort clusters", fill: C.tealPale, color: C.teal },
    { x: 3.18, w: 2.72, text: "spikes plotted by median probe depth", fill: C.goldPale, color: "8A6515" },
    { x: 6.08, w: 2.48, text: "illustrative pair · uncorrected screen", fill: C.coralPale, color: C.coral },
  ];
  for (const tag of tags) {
    slide.addShape(pptx.ShapeType.roundRect, {
      x: tag.x, y: 5.50, w: tag.w, h: 0.44, rectRadius: 0.06,
      fill: { color: tag.fill }, line: { color: tag.fill },
    });
    slide.addText(tag.text, {
      x: tag.x + 0.08, y: 5.61, w: tag.w - 0.16, h: 0.20,
      fontFace: FONT, fontSize: 9.3, bold: true, color: tag.color,
      align: "center", margin: 0, fit: "shrink",
    });
  }
}
{
  const slide = await addSingleImageSlide(4, paths.probeBoundary);
  // The source plot reserves x > 1 for run-level annotations. Hide only the
  // unused axis extension so the normalized 0-1 coordinate cannot be misread.
  slide.addShape(pptx.ShapeType.rect, {
    x: 6.32, y: 5.68, w: 1.91, h: 0.36,
    fill: { color: C.white }, line: { color: C.white },
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 6.43, y: 5.68, w: 0, h: 0.08,
    line: { color: "111111", width: 0.8 },
  });
  slide.addText("1.0", {
    x: 6.27, y: 5.80, w: 0.32, h: 0.15, fontFace: FONT, fontSize: 9.2,
    color: "111111", align: "center", margin: 0,
  });
}
for (let idx = 0; idx < patientDepthPanels.length; idx += 1) {
  const slideIndex = 5 + idx;
  const panel = patientDepthPanels[idx];
  await addSingleImageSlide(slideIndex, paths.patientDepthImages[idx], {
    replaceEmbeddedTitle: `${panel.subject}: unit/run responses across probe depth`,
  });
}
await addSingleImageSlide(10, paths.timeGeneralization, {
  replaceEmbeddedTitle: "Time-resolved and cross-temporal construction decoding",
});
await addSingleImageSlide(11, paths.decoderSensitivity);
await addSingleImageSlide(12, paths.populationDiagnostics);
await addSingleImageSlide(13, paths.baselineState, {
  replaceEmbeddedTitle: "Baseline-state controls",
});
await addSingleImageSlide(14, paths.depthArtifact, {
  replaceEmbeddedTitle: "Relative-depth organization and IED/artifact controls",
});
await addSingleImageSlide(15, paths.stateSpace, {
  replaceEmbeddedTitle: "Population state-space and representational geometry",
});
{
  const slide = await addSingleImageSlide(16, paths.clusterComposition);
  slide.addShape(pptx.ShapeType.rect, {
    x: 1.55, y: 2.18, w: 6.70, h: 0.30,
    fill: { color: C.white }, line: { color: C.white },
  });
  slide.addText("Sessions occupy different subsets of six possible response archetypes", {
    x: 1.61, y: 2.23, w: 6.58, h: 0.20, fontFace: FONT, fontSize: 12.2,
    bold: true, color: C.ink, margin: 0, align: "left", fit: "shrink",
  });
  slide.addText("no qualifying entries", {
    x: 3.63, y: 4.19, w: 1.50, h: 0.17, fontFace: FONT, fontSize: 8.3,
    italic: true, color: C.gray, margin: 0, align: "center",
  });
}

// S17 - two prior summary figures combined as complementary panels.
{
  const index = 17;
  const slide = pptx.addSlide();
  addBase(slide, index, legends[index - 1]);
  slide.addText("a", { x: 0.66, y: 0.91, w: 0.25, h: 0.24, fontFace: FONT, fontSize: 13, bold: true, color: C.ink, margin: 0 });
  await addContainedImage(slide, paths.nounScreen, { x: 0.78, y: 0.88, w: 5.30, h: 2.73 }, "Matched-noun slot unit census");
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 6.24, y: 1.35, w: 2.16, h: 1.55, rectRadius: 0.06,
    fill: { color: C.tealPale }, line: { color: C.tealPale },
  });
  slide.addText("279 / 1,782", {
    x: 6.47, y: 1.66, w: 1.70, h: 0.42, fontFace: FONT, fontSize: 21,
    bold: true, color: C.teal, align: "center", margin: 0,
  });
  slide.addText("screen-positive\nunit/run entries", {
    x: 6.47, y: 2.15, w: 1.70, h: 0.42, fontFace: FONT, fontSize: 9.5,
    color: C.ink, align: "center", margin: 0,
  });
  slide.addText("b", { x: 0.66, y: 3.82, w: 0.25, h: 0.24, fontFace: FONT, fontSize: 13, bold: true, color: C.ink, margin: 0 });
  await addContainedImage(slide, paths.nounClassControl, { x: 0.78, y: 3.79, w: 5.30, h: 2.73 }, "Noun-class replication control");
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 6.24, y: 4.26, w: 2.16, h: 1.55, rectRadius: 0.06,
    fill: { color: C.goldPale }, line: { color: C.goldPale },
  });
  slide.addText("33 / 279", {
    x: 6.47, y: 4.57, w: 1.70, h: 0.42, fontFace: FONT, fontSize: 21,
    bold: true, color: "8A6515", align: "center", margin: 0,
  });
  slide.addText("same direction in\nboth noun strata", {
    x: 6.47, y: 5.06, w: 1.70, h: 0.42, fontFace: FONT, fontSize: 9.5,
    color: C.ink, align: "center", margin: 0,
  });
}

await addSingleImageSlide(18, paths.subjectHeatmap);
await addSingleImageSlide(19, paths.postMarkerHeatmap);
await addSingleImageSlide(20, paths.stimulusIdentifiability);
await addSingleImageSlide(21, paths.positionControls);
{
  const slide = await addSingleImageSlide(22, paths.alignmentSpecs);
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 2.05, y: 6.08, w: 5.16, h: 0.42, rectRadius: 0.05,
    fill: { color: C.tealPale }, line: { color: C.tealPale },
  });
  slide.addText("All patient max-statistic FWER P values ≥ 0.0625", {
    x: 2.23, y: 6.18, w: 4.80, h: 0.20, fontFace: FONT, fontSize: 11.0,
    bold: true, color: C.teal, align: "center", margin: 0,
  });
}
await addSingleImageSlide(23, paths.testSensitivity);
await addSingleImageSlide(24, paths.positionValidation);

// Supplementary Table S1 - editable patient-level recording summary.
{
  const slide = pptx.addSlide();
  slide.background = { color: C.cream };
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.44, y: 0.18, w: 2.56, h: 0.36, rectRadius: 0.06,
    fill: { color: C.teal }, line: { color: C.teal },
  });
  slide.addText("SUPPLEMENTARY TABLE S1", {
    x: 0.56, y: 0.245, w: 2.32, h: 0.20, fontFace: FONT, fontSize: 9.3,
    bold: true, color: C.white, align: "center", margin: 0, charSpacing: 0.8,
  });
  slide.addText("Single-neuronal encoding of syntactic computations", {
    x: 6.04, y: 0.255, w: 6.84, h: 0.18, fontFace: FONT, fontSize: 8.5,
    color: C.gray, align: "right", margin: 0,
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.42, y: 0.68, w: 12.46, h: 6.42, rectRadius: 0.05,
    fill: { color: C.white }, line: { color: C.line, width: 0.8 }, shadow: shadow(),
  });
  slide.addText("RECORDING AND TASK SUMMARY", {
    x: 0.82, y: 0.98, w: 3.30, h: 0.24, fontFace: FONT, fontSize: 9.2,
    bold: true, color: C.teal, margin: 0, charSpacing: 0.8,
  });
  slide.addText("Patient-level recording and task summary", {
    x: 0.82, y: 1.30, w: 7.50, h: 0.45, fontFace: FONT, fontSize: 25.0,
    bold: true, color: C.ink, margin: 0, fit: "shrink",
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 9.09, y: 1.20, w: 3.34, h: 0.64, rectRadius: 0.05,
    fill: { color: C.tealPale }, line: { color: C.tealPale },
  });
  slide.addText("5 patients  ·  8 runs  ·  680 trials", {
    x: 9.28, y: 1.42, w: 2.96, h: 0.21, fontFace: FONT, fontSize: 10.6,
    bold: true, color: C.teal, align: "center", margin: 0, fit: "shrink",
  });

  const headers = ["Patient", "Usable\nruns", "Task duration\n(min)", "Task\ntrials", "Unit\nentries", "Good", "MUA"];
  const records = [
    ["gz001", "1", "5.17", "85", "63", "20", "43"],
    ["gz002", "1", "5.12", "85", "204", "119", "85"],
    ["gz003", "2", "10.26", "170", "131", "75", "56"],
    ["gz004", "2", "10.21", "170", "73", "46", "27"],
    ["gz005", "2", "10.21", "170", "248", "201", "47"],
    ["Total", "8", "40.97", "680", "719", "461", "258"],
  ];
  const tableRows = [
    headers.map((text, colIdx) => ({
      text,
      options: {
        fill: { color: C.teal }, color: C.white, bold: true,
        align: colIdx === 0 ? "left" : "center", valign: "mid",
        margin: colIdx === 0 ? [0.08, 0.13, 0.08, 0.16] : [0.08, 0.08, 0.08, 0.08],
      },
    })),
    ...records.map((row, rowIdx) => row.map((text, colIdx) => {
      const isTotal = rowIdx === records.length - 1;
      return {
        text,
        options: {
          fill: { color: isTotal ? C.deep : (rowIdx % 2 === 0 ? "FFFFFF" : "F1F6F4") },
          color: isTotal ? C.white : C.ink,
          bold: isTotal || colIdx === 0,
          align: colIdx === 0 ? "left" : "center", valign: "mid",
          margin: colIdx === 0 ? [0.08, 0.13, 0.08, 0.16] : [0.08, 0.08, 0.08, 0.08],
        },
      };
    })),
  ];
  slide.addTable(tableRows, {
    x: 0.82, y: 2.05, w: 11.69, h: 3.30,
    colW: [2.05, 1.40, 1.82, 1.60, 1.82, 1.50, 1.50],
    rowH: [0.58, 0.45, 0.45, 0.45, 0.45, 0.45, 0.52],
    fontFace: FONT, fontSize: 13.1,
    border: { type: "solid", pt: 0.7, color: C.line },
    margin: 0,
  });

  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.82, y: 5.65, w: 11.69, h: 0.98, rectRadius: 0.04,
    fill: { color: C.legend }, line: { color: C.line, width: 0.7 },
  });
  slide.addText([
    { text: "Table note. ", options: { bold: true, color: C.teal } },
    { text: "Task duration and task trials summarize the auditory active/passive sentence task. Unit entries are run-specific Kilosort clusters in the 719-entry main unit-level analysis table: 461 entries carried the Kilosort good label and 258 carried the multi-unit activity (MUA) label. This analysis table is distinct from the broader 1,782-entry registered unit/run census in Supplementary Fig. S2. Patients (n = 5) are independent biological units; runs and unit entries are nested within patients." },
  ], {
    x: 1.05, y: 5.89, w: 11.23, h: 0.51, fontFace: FONT, fontSize: 10.8,
    color: C.deep, margin: 0, fit: "shrink", valign: "mid", breakLine: false,
  });
}

await pptx.writeFile({ fileName: OUT });
// PptxGenJS emits notesMasterIdLst after sldIdLst. Microsoft Office accepts
// that ordering, but the ECMA schema requires it before sldIdLst.
{
  const zip = await JSZip.loadAsync(await fsp.readFile(OUT));
  const presentation = await zip.file("ppt/presentation.xml").async("string");
  const notesMaster = presentation.match(/<p:notesMasterIdLst>[\s\S]*?<\/p:notesMasterIdLst>/)?.[0];
  if (notesMaster) {
    const reordered = presentation
      .replace(notesMaster, "")
      .replace("<p:sldIdLst>", `${notesMaster}<p:sldIdLst>`);
    zip.file("ppt/presentation.xml", reordered);
    await fsp.writeFile(
      OUT,
      await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }),
    );
  }
}
await fsp.writeFile(
  MANIFEST,
  `${JSON.stringify({
    output: OUT,
    created_at: new Date().toISOString(),
    historical_layout_date: "2026-08-10",
    status: "Historical base layout; does not assemble the final renumbered supplementary deck",
    manuscript: path.join(ROOT, "manuscript", "Manuscript V1-lht(1).docx"),
    main_figures: [1, 2, 3, 4].map((n) => path.join(ROOT, "manuscript", `Fig${n}.pdf`)),
    supplementary_figures: legends.map((legend, idx) => ({
      number: `S${idx + 1}`,
      title: legend.title,
      source_assets: legend.sources,
    })),
    supplementary_tables: [{
      number: "S1",
      title: "Patient-level recording and task summary",
      source_data: TABLE_S1_SOURCE,
      scope_note: "719-entry main unit-level analysis table; distinct from the broader 1,782-entry registry in Supplementary Fig. S2",
    }],
  }, null, 2)}\n`,
  "utf8",
);

console.log(OUT);
