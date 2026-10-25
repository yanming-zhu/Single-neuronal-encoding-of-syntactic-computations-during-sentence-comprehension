# Code archive validation — 2026-10-04

Author: Yanming Zhu

This report supersedes the 2026-08-09 packaging-validation statement. It concerns this updated code archive, not a full rerun of the scientific study.

## Completed checks

- Manifest: 121 canonical code entries; every listed archive path exists; all SHA-256 values match; no unlisted Python/JavaScript code.
- Syntax: 120 unique Python files compile successfully; one JavaScript builder passes `node --check`.
- Navigation: 101 internal symlinks resolve within the archive.
- Static local-import closure: 1,226 explicit import statements inspected; no missing package-local module found. Dynamic imports are outside this check.
- Regression tests: **28 passed, 2 subtests passed** in a relocated temporary copy using Python 3.10.18 and the versions in `requirements-test.txt`.
- Tests cover predictive-information calculation, trial-wise averaging of repeated out-of-fold probabilities, held-pair separation, training-only selection, fixed/nested regularization, patient-level sign tests, corrected scrambled-verb character targets, explicit historical/corrected routing, data-derived counts/comparisons, stale comparison-table rejection by avoidance, relocation, and packaging failures.
- Fig. 3H smoke export from copied frozen inputs succeeded: totals 172/279/148; union 425; triple intersection 26; common 1,782-unit universe. No unit screen was rerun.
- Fig. 1H/I smoke exports from copied frozen AUC matrices succeeded: seven sessions/five participants, 51 time points, 51×51 cross-time matrix, finite transformations, no clipping of displayed summary values, inverse transforms agree with the saved AUCs.
- Both Fig. 1 PDF assembly scripts ran successfully in the temporary copy. Pixels outside replaced panels were unchanged; H assembly preserved the previously replaced I panel.
- Verb timing loaders were checked against temporary copies of the existing historical and corrected tables: 72 (30/18/24) and 77 (31/18/28) timing entries, respectively, each drawn from 719 analyzed entries. These counts were read from existing tables, not obtained from a new raw-data analysis.
- Revised verb CLI entry points and supplementary Node `--help` passed.

## Environments actually exercised

Core tests: Python 3.10.18, NumPy 1.26.4, pandas 2.3.1, SciPy 1.15.3, Matplotlib 3.10.0, scikit-learn 1.7.1, Pillow 11.3.0, networkx 3.4.2, pytest 9.1.1.

Frozen-input panel exports: Python 3.11, NumPy 1.23.5, pandas 2.1.2, SciPy 1.11.3, Matplotlib 3.8.1, matplotlib-venn 0.11.10, with locally available Arial. Fontconfig emitted a cache-location warning, but all export processes exited successfully and their numerical/geometry assertions passed.

PDF assembly: Python 3.12 ARM64 runtime with pypdf 6.10.0 and pypdfium2 5.13.0. Node syntax/help checks do not constitute a complete PPTX build. Node dependency versions in `package.json` were verified as available in the local runtime; no fresh npm install was performed.

These are multiple existing environments, not a demonstration that every optional dependency has been installed together in a new environment. The validator reports missing external packages separately as skips for the interpreter used to invoke it.

## Not performed

- No complete raw-recording extraction, Kilosort re-sorting, all-unit neural reanalysis, model-weight extraction or final supplementary-deck rebuild.
- No claim that the original Kilosort version/settings/hardware have been recovered for every run.
- No independent review of the corrected audio anchors or manuscript adoption of the corrected three-condition result.
- No changes to the original manuscript, final figure files or existing analysis results. Generated smoke-test figures remain in a disposable temporary copy.

For reproducible checking in another installation, use:

```sh
python scripts/validate_code_package.py --output /tmp/tan_code_validation.json
python -m pytest -q tests
```

Consult README for required external inputs, analysis versions and scientific interpretation limits. The historical supplementary manifest/gap-audit files describe earlier figure layouts and do not override this validation record.
