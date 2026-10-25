"""Replot the existing time-resolved decoding data on the signed d_A scale."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import norm

SOURCE = ROOT / 'neuropixel2025/data/combined_active_passive_time_generalization/active_passive_time_generalization_matrices.npz'
FIGURES = OUT / 'figures'
DATA = OUT / 'source_data'
for folder in (FIGURES, DATA):
    folder.mkdir(parents=True, exist_ok=True)

# Figure contract: the median describes a transient construction-related
# discriminability peak; the interquartile band shows session variability.
# This quantitative panel stays in the supplied mixed-modality composite.
# Export vector text, complete source data, and untrimmed session values.
payload = np.load(SOURCE, allow_pickle=True)
centers = np.asarray(payload['centers_sec'], dtype=float)
auc = np.asarray(payload['diag_auc'], dtype=float)
assert auc.shape == (7, 51)
assert np.allclose(auc, np.diagonal(payload['auc_matrices'], axis1=1, axis2=2))
assert np.isfinite(auc).all() and ((auc > 0) & (auc < 1)).all()
d = np.sqrt(2) * norm.ppf(auc)
median = np.median(d, axis=0)
q25, q75 = np.quantile(d, [.25, .75], axis=0)
median_auc = np.median(auc, axis=0)
assert np.allclose(norm.cdf(d / np.sqrt(2)), auc, atol=1e-14)
assert np.allclose(median, np.sqrt(2) * norm.ppf(median_auc))
peak = int(np.argmax(median))
assert np.isclose(centers[peak], .30) and np.isclose(median_auc[peak], .67)
assert len(set(payload['subjects'])) == 5

rows = []
for subject, run, a, values in zip(payload['subjects'], payload['runs'], auc, d):
    rows.append(pd.DataFrame({'participant': str(subject), 'session': str(run),
                             'time_sec': centers, 'held_out_auc': a, 'd_A': values}))
pd.concat(rows, ignore_index=True).to_csv(DATA / 'Fig1h_session_discriminability.csv', index=False)
pd.DataFrame({'time_sec': centers, 'median_auc': median_auc, 'median_d_A': median,
              'session_d_A_q25': q25, 'session_d_A_q75': q75,
              'n_sessions': 7, 'n_participants': 5}).to_csv(
                  DATA / 'Fig1h_median_discriminability.csv', index=False)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7.2, 'axes.labelsize': 7.2, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'axes.linewidth': .6,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
    'text.color': '#17322F', 'axes.labelcolor': '#17322F',
    'axes.edgecolor': '#17322F', 'xtick.color': '#17322F', 'ytick.color': '#17322F',
})
width_pt, height_pt = 102, 129
fig = plt.figure(figsize=(width_pt / 72, height_pt / 72), facecolor='none')
# The transparent left notch preserves the neighbouring heatmap's Hz label.
fig.add_artist(Rectangle((5/width_pt, 0), 97/width_pt, 1,
                         transform=fig.transFigure, facecolor='white', edgecolor='none', zorder=-1))
fig.add_artist(Rectangle((0, 110/height_pt), 1, 19/height_pt,
                         transform=fig.transFigure, facecolor='white', edgecolor='none', zorder=-1))
ax = fig.add_axes([30/width_pt, 26/height_pt, 55/width_pt, 81/height_pt])
ax.fill_between(centers, q25, q75, color='#BCC5D0', alpha=.50,
                linewidth=0, zorder=1)
ax.axhline(0, color='#5D6E6B', linewidth=.65, linestyle=(0, (2, 2)), zorder=2)
ax.plot(centers, median, color='#377EB8', linewidth=1.25, zorder=3)
ax.plot(centers[peak], median[peak], marker='o', markersize=2.6,
        color='#377EB8', linewidth=0, zorder=4)
ax.set_xlim(centers[0], centers[-1])
ax.set_ylim(-1.2, 1.2)
ax.set_xticks([-.5, .5, 1.5])
ax.set_yticks([-1, 0, 1])
ax.tick_params(pad=1.5)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('Time (s)', labelpad=2)
fig.text(12/width_pt, 66.5/height_pt, r'Discriminability ($d_A$)', rotation=90,
         ha='center', va='center', fontsize=7.2)
fig.text(2/width_pt, 128/height_pt, 'h', va='top', ha='left',
         color='black', fontsize=11, fontweight='bold')
fig.text(58/width_pt, 128/height_pt, 'Top-5 decoder', ha='center', va='top', fontsize=7.2)
fig.text(58/width_pt, 117/height_pt, 'Peak 0.62 at 0.30 s', ha='center', va='top', fontsize=6.1)

stem = FIGURES / 'Fig1h_time_resolved_discriminability'
for ext in ('pdf', 'svg', 'png', 'tiff'):
    kw = {'dpi': 600, 'facecolor': 'none' if ext in ('pdf', 'svg') else 'white'}
    if ext == 'tiff': kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(stem.with_suffix('.' + ext), **kw)
plt.close(fig)

report = {
    'figure_contract': {
        'claim': 'Session-median discriminability shows a descriptive peak at 0.30 s after function word onset.',
        'evidence': 'Existing held-out diagonal AUC values from seven sessions in five participants; no refitting or significance test.',
        'archetype': 'Quantitative time series in an existing mixed-modality composite',
        'backend': 'Python matplotlib',
        'statistics': 'Blue session median and grey interquartile range; descriptive variability, not a confidence interval or significance test.',
        'exports': 'Editable panel SVG/PDF and 600 dpi PNG/TIFF; full-figure PDF, PNG and TIFF; session and median CSVs.',
    },
    'formula': 'd_A = sqrt(2) * Phi_inverse(AUC)',
    'aggregation': 'Transform each session diagonal, then take the median across seven sessions at each timepoint.',
    'chance_level': 0, 'negative_values_retained': True,
    'n_sessions': 7, 'n_participants': 5, 'timepoints': len(centers),
    'session_auc_range': [float(auc.min()), float(auc.max())],
    'session_d_A_range': [float(d.min()), float(d.max())],
    'median_peak_time_sec': float(centers[peak]),
    'median_peak_auc': float(median_auc[peak]),
    'median_peak_d_A': float(median[peak]),
    'y_axis_limits': [-1.2, 1.2],
    'displayed_summary': 'Median and 25th-75th percentiles after session-wise conversion to d_A.',
    'interquartile_band_range': [float(q25.min()), float(q75.max())],
    'summary_values_clipped': int(((q25 < -1.2) | (q75 > 1.2)).sum()),
    'individual_session_values': 'All untrimmed session values are exported in source_data/Fig1h_session_discriminability.csv; the main panel displays the session median and IQR.',
    'all_values_finite': bool(np.isfinite(d).all()),
    'inverse_transform_matches_original_auc': True,
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'placement_pdf_points': {'left': 403, 'bottom': 157, 'width': width_pt, 'height': height_pt},
    'highlight': 'Earlier orange time band removed; a dot identifies the descriptive maximum, with no significance symbol.',
}
original_stats = pd.read_csv(SOURCE.parent / 'active_passive_time_resolved_diagonal_time_stats.csv')
original_peak = original_stats.loc[np.isclose(original_stats['time_sec'], centers[peak])].iloc[0]
report['original_statistical_audit'] = {
    'unit_of_analysis': 'Seven sessions, not five independent participants',
    'peak_time_sec': float(centers[peak]),
    'one_sided_wilcoxon_p_greater': float(original_peak['wilcoxon_p_greater']),
    'bh_fdr_q': float(original_peak['q_bh']),
    'minimum_uncorrected_timepoint_p': float(original_stats['wilcoxon_p_greater'].min()),
    'significant_timepoints_fdr_0p05': int((original_stats['q_bh'] < .05).sum()),
    'new_significance_test_performed': False,
}
(OUT / 'audit/figure_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
