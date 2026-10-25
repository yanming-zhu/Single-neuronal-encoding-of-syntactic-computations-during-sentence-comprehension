"""Replot the descriptive cross-time matrix on the signed d_A scale."""
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
import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib.patches import Rectangle

SOURCE = ROOT / 'neuropixel2025/data/combined_active_passive_time_generalization'
FIGURES = OUT / 'figures'
DATA = OUT / 'source_data'
for folder in (FIGURES, DATA):
    folder.mkdir(parents=True, exist_ok=True)

payload = np.load(SOURCE / 'active_passive_time_generalization_matrices.npz', allow_pickle=True)
centers = np.asarray(payload['centers_sec'], dtype=float)
matrices = np.asarray(payload['auc_matrices'], dtype=float)
assert matrices.shape == (7, 51, 51)
assert np.allclose(np.diagonal(matrices, axis1=1, axis2=2), payload['diag_auc'])
median_auc = np.median(matrices, axis=0)
assert np.isfinite(median_auc).all() and ((median_auc > 0) & (median_auc < 1)).all()
discriminability = np.sqrt(2) * norm.ppf(median_auc)
assert np.allclose(norm.cdf(discriminability / np.sqrt(2)), median_auc, atol=1e-14)
assert np.allclose(discriminability, np.median(np.sqrt(2) * norm.ppf(matrices), axis=0))
assert np.max(np.abs(discriminability)) < 1

earlier = ROOT / 'reports/nature_science_draft/assembled_figures_20260805/subplots/source_data/time_generalization_d_e_stacked_30x100mm_median_matrix_long.csv'
old_export = pd.read_csv(earlier)
assert np.allclose(old_export['median_auc'].to_numpy().reshape(51, 51), median_auc)

train, test = np.meshgrid(centers, centers, indexing='ij')
pd.DataFrame({
    'train_time_sec': train.ravel(), 'test_time_sec': test.ravel(),
    'median_auc': median_auc.ravel(), 'd_A': discriminability.ravel(),
    'n_sessions': 7, 'n_participants': 5,
}).to_csv(DATA / 'Fig1i_cross_temporal_discriminability.csv', index=False)
run_rows = []
for subject, run, matrix in zip(payload['subjects'], payload['runs'], matrices):
    run_rows.append(pd.DataFrame({
        'participant': str(subject), 'session': str(run),
        'train_time_sec': train.ravel(), 'test_time_sec': test.ravel(),
        'held_out_auc': matrix.ravel(),
    }))
pd.concat(run_rows, ignore_index=True).to_csv(DATA / 'Fig1i_session_auc_source.csv', index=False)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7.2, 'axes.labelsize': 7.2, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
    'text.color': '#17322F', 'axes.labelcolor': '#17322F',
    'axes.edgecolor': '#17322F', 'xtick.color': '#17322F', 'ytick.color': '#17322F',
})

# The panel occupies the same region of the supplied composite. Its full range
# is shown using a symmetric, linear d_A colour scale with a neutral zero.
width_pt, height_pt = 102, 132
fig = plt.figure(figsize=(width_pt / 72, height_pt / 72), facecolor='none')
# Leave the narrow strip belonging to the neighbouring panel untouched.
fig.add_artist(Rectangle((5/width_pt, 0), 97/width_pt, 1,
                         transform=fig.transFigure, facecolor='white', edgecolor='none', zorder=-1))
fig.add_artist(Rectangle((0, 110/height_pt), 1, 22/height_pt,
                         transform=fig.transFigure, facecolor='white', edgecolor='none', zorder=-1))
ax = fig.add_axes([30/width_pt, 51/height_pt, 55/width_pt, 59/height_pt])
im = ax.imshow(
    discriminability, extent=[centers[0], centers[-1], centers[-1], centers[0]],
    origin='upper', aspect='auto', interpolation='nearest', cmap='PuOr_r',
    vmin=-1, vmax=1, rasterized=True,
)
ax.plot([centers[0], centers[-1]], [centers[0], centers[-1]],
        color='#4E4E4E', linewidth=0.55, linestyle=(0, (2.0, 1.6)))
ax.set_xticks([-0.5, 0.5, 1.5])
ax.set_yticks([-0.5, 0.5, 1.5])
ax.tick_params(pad=1.5)
ax.set_xlabel('Test (s)', labelpad=2)
fig.text(12/width_pt, 80.5/height_pt, 'Train (s)', rotation=90,
         ha='center', va='center', fontsize=7.2)
fig.text(2/width_pt, 131/height_pt, 'i', va='top', ha='left',
         color='black', fontsize=11, fontweight='bold')
fig.text(58/width_pt, 131/height_pt, 'Cross-time\ndiscriminability',
         ha='center', va='top', fontsize=7.2, linespacing=1.03)
cbax = fig.add_axes([30/width_pt, 11/height_pt, 55/width_pt, 4/height_pt])
cb = fig.colorbar(im, cax=cbax, orientation='horizontal', ticks=[-1, 0, 1])
cb.outline.set_linewidth(0.5)
cb.ax.tick_params(length=2, width=0.5, pad=1, labelsize=7)
cb.ax.set_xticklabels(['−1', '0', '1'])
fig.text(22/width_pt, 13/height_pt, r'$d_A$', va='center', ha='right', fontsize=8)
fig.text(57.5/width_pt, 22/height_pt, '0 = chance', va='bottom', ha='center', fontsize=6.2)

stem = FIGURES / 'Fig1i_cross_temporal_discriminability'
for ext in ('pdf', 'svg', 'png', 'tiff'):
    kw = {'dpi': 600, 'facecolor': 'none' if ext in ('pdf', 'svg') else 'white'}
    if ext == 'tiff': kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(stem.with_suffix('.' + ext), **kw)
plt.close(fig)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

report = {
    'figure_contract': {
        'claim': 'The descriptive cross-temporal matrix characterizes limited generalization of construction-related discriminability across time.',
        'evidence': 'Existing held-out AUC matrices from seven sessions in five participants; no decoder refitting or new significance test.',
        'archetype': 'Quantitative heatmap within an existing mixed-modality composite',
        'backend': 'Python matplotlib',
        'statistics': 'Cellwise session median, descriptive; sessions are not treated as independent participants.',
        'exports': 'Editable panel SVG/PDF and 600 dpi PNG/TIFF; revised full-figure PDF and PNG.',
    },
    'formula': 'd_A = sqrt(2) * Phi_inverse(cellwise median session AUC)',
    'chance_level': 0,
    'negative_values_retained': True,
    'n_sessions': 7,
    'n_participants': len(set(payload['subjects'])),
    'matrix_shape': list(discriminability.shape),
    'matrix_auc_range': [float(median_auc.min()), float(median_auc.max())],
    'matrix_d_A_range': [float(discriminability.min()), float(discriminability.max())],
    'time_range_sec': [float(centers[0]), float(centers[-1])],
    'color_scale': [-1, 1],
    'colormap': 'PuOr_r',
    'n_cells_clipped': int((np.abs(discriminability) > 1).sum()),
    'all_values_finite': True,
    'inverse_transform_matches_original_auc': True,
    'matches_original_panel_source_export': True,
    'peak_highlight_removed': 'The earlier 0.20-0.40 s orange box was descriptive, not a significance mask; no selective highlighting is used.',
    'diagonal_line': 'Dashed grey reference marks equal training and testing times.',
    'source_sha256': {p.name: sha(p) for p in [SOURCE / 'active_passive_time_generalization_matrices.npz', earlier]},
    'placement_pdf_points': {'left': 403, 'bottom': 25, 'width': width_pt, 'height': height_pt},
}
(OUT / 'audit/figure_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
