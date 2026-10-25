from pathlib import Path
import sys, json, itertools, hashlib
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import norm, binomtest
from scripts import make_fig2_population_180mm_revised as source
import matplotlib.pyplot as plt

OUT = ROOT / 'manuscript/revision_20260913'
FIG = OUT / 'figures'
DATA = OUT / 'source_data'
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
input_path = ROOT / 'neuropixel2025/data/combined_unit_population_predictive_information/patient_values.csv'
frame = pd.read_csv(input_path)
assert frame.auc.between(0, 1, inclusive='neither').all()
frame['d_A'] = np.sqrt(2) * norm.ppf(frame.auc)
frame.to_csv(DATA / 'discriminability_patient_values.csv', index=False)

def signflip(v):
    v = np.asarray(v, float)
    null = [np.mean(v * s) for s in itertools.product([-1, 1], repeat=len(v))]
    return float(np.mean(np.asarray(null) >= np.mean(v) - 1e-12))

stats = {}
for reg in ['fixed', 'nested']:
    for validation in ['all_sentences', 'unseen_sentence_pair']:
        sub = frame[(frame.regularization == reg) & (frame.validation == validation)]
        pivot = sub.pivot(index='subject', columns='model', values='d_A')
        gain = pivot.top5_population - pivot.single_unit
        stats[f'{reg}_{validation}'] = {
            'single_d_A': float(pivot.single_unit.median()),
            'population_d_A': float(pivot.top5_population.median()),
            'paired_gain_median': float(gain.median()),
            'n_positive': int((gain > 0).sum()),
            'n_patients': len(gain),
            'exact_sign_p_one_sided': float(binomtest(int((gain > 0).sum()), len(gain), alternative='greater').pvalue),
        }

ablation = source.load_top_unit_ablation_data()['patient_values'].copy()
ablation['ranked_d_A'] = np.sqrt(2) * norm.ppf(ablation.ranked_removal_auc)
ablation['random_d_A'] = np.sqrt(2) * norm.ppf(ablation.random_removal_median_auc)
ablation['delta_d_A'] = ablation.random_d_A - ablation.ranked_d_A
ablation.to_csv(DATA / 'ablation_patient_discriminability.csv', index=False)
stats['ablation'] = {}
for n, sub in ablation.groupby('remove_label'):
    stats['ablation'][str(n)] = {
        'ranked_median': float(sub.ranked_d_A.median()),
        'random_median': float(sub.random_d_A.median()),
        'paired_gain_median': float(sub.delta_d_A.median()),
        'exact_signflip_p_one_sided': signflip(sub.delta_d_A),
    }

_, sensitivity = source.load_sensitivity_data()
for col in ['nested_single_unit_auc', 'descriptive_best_population_auc']:
    sensitivity[col.replace('auc', 'd_A')] = np.sqrt(2) * norm.ppf(sensitivity[col])
sensitivity.to_csv(DATA / 'sensitivity_patient_discriminability.csv', index=False)
stats['sensitivity_population_medians'] = sensitivity.groupby('spec_order').descriptive_best_population_d_A.median().tolist()

def pair_panel(ax, validation, letter=None):
    sub = frame[(frame.regularization == 'fixed') & (frame.validation == validation)]
    pivot = sub.pivot(index='subject', columns='model', values='d_A').sort_index()
    for subject, row in pivot.iterrows():
        ax.plot([0, 1], [row.single_unit, row.top5_population], color='#a8b1ba', lw=.85, zorder=1)
    ax.scatter(np.zeros(5), pivot.single_unit, s=23, facecolor='white', edgecolor='#505b65', lw=.9, zorder=3)
    ax.scatter(np.ones(5), pivot.top5_population, s=25, color='#2171b5', edgecolor='white', lw=.4, zorder=3)
    for x, val in enumerate([pivot.single_unit.median(), pivot.top5_population.median()]):
        ax.plot([x-.14, x+.14], [val, val], color='#172b40', lw=2.0, zorder=4)
    ax.axhline(0, color='#9099a2', lw=.7, ls='--')
    ax.set(xlim=(-.3,1.3), ylim=(-.86,2.20), xticks=[0,1], xticklabels=['Single unit','Top-5\npopulation'], ylabel=r'Discriminability, $d_A$')
    title = 'Within-material decoding' if validation == 'all_sentences' else 'Unseen sentence pairs'
    ax.set_title(title, loc='left', weight='bold', fontsize=8, pad=5)
    s = stats[f'fixed_{validation}']
    ax.text(.02,.98,f"Median: {s['single_d_A']:.2f} → {s['population_d_A']:.2f}\nPaired gain: +{s['paired_gain_median']:.2f}; P = 0.1875",transform=ax.transAxes,va='top',fontsize=7)
    if letter: source.panel_label(ax,letter)

def gain_panel(ax):
    for k, validation in enumerate(['all_sentences','unseen_sentence_pair']):
        sub=frame[(frame.regularization=='fixed') & (frame.validation==validation)]
        p=sub.pivot(index='subject',columns='model',values='d_A')
        gain=p.top5_population-p.single_unit
        ax.scatter(np.full(5,k)+np.linspace(-.07,.07,5),gain,s=27,color='#2171b5',edgecolor='white',lw=.5)
        ax.plot([k-.19,k+.19],[gain.median()]*2,color='#172b40',lw=2)
    ax.axhline(0,color='#9099a2',ls='--',lw=.8)
    ax.set(xticks=[0,1],xticklabels=['Within\nmaterial','Unseen\npairs'],ylabel=r'Population gain, $\Delta d_A$',ylim=(-.24,2.52),xlim=(-.4,1.4))
    ax.set_title('Within-patient difference',loc='left',weight='bold',fontsize=8)
    ax.text(.02,.98,'4/5 patients positive in each test\nExact one-sided sign P = 0.1875',transform=ax.transAxes,va='top',fontsize=7)
    source.panel_label(ax,'c')

def save(fig, name):
    for ext in ['pdf','svg','png']:
        fig.savefig(FIG/f'{name}.{ext}',dpi=400,facecolor='white')
    plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(180/25.4,74/25.4))
fig.subplots_adjust(left=.075,right=.987,bottom=.24,top=.88,wspace=.5)
pair_panel(axes[0],'all_sentences','a')
pair_panel(axes[1],'unseen_sentence_pair','b')
gain_panel(axes[2])
fig.text(.075,.035,'Points: five patients; bars: medians. Both readouts use training-fold unit selection and fixed L2 C = 0.1.',fontsize=7)
save(fig,'Discriminability_comparison')

# Rebuild the complete Fig. 2, retaining the original source data for panels c, f and g.
fig=plt.figure(figsize=(180/25.4,176/25.4),facecolor='white')
grid=fig.add_gridspec(3,6,height_ratios=[1.04,.90,1.22],left=.075,right=.978,bottom=.077,top=.95,wspace=1.05,hspace=.78)
pair_panel(fig.add_subplot(grid[0,0:2]),'all_sentences','a')
pair_panel(fig.add_subplot(grid[0,2:4]),'unseen_sentence_pair','b')
source.draw_weight_depth(fig.add_subplot(grid[0,4:6]),source.load_weight_data())
ax=fig.add_subplot(grid[1,0:3]);source.panel_label(ax,'d',x=-.14,y=1.09)
for _,row in sensitivity.iterrows():
    x=float(row.spec_order)
    ax.plot([x-.11,x+.11],[row.nested_single_unit_d_A,row.descriptive_best_population_d_A],color=source.PATIENT_COLORS[row.subject],lw=.7,marker='o',ms=2.8,alpha=.7)
med=sensitivity.groupby('spec_order')[['nested_single_unit_d_A','descriptive_best_population_d_A']].median()
ax.plot(med.index-.11,med.nested_single_unit_d_A,color='#606870',marker='o',ms=3.5,lw=1.2,label='Single unit')
ax.plot(med.index+.11,med.descriptive_best_population_d_A,color='#2171b5',marker='o',ms=3.5,lw=1.2,label='Selected best population')
ax.axhline(0,color='#9099a2',lw=.7,ls='--')
ax.set(xticks=range(4),xticklabels=['All units\nall trials','All units\nclean','Good units\nall trials','Good units\nclean'],ylabel=r'Discriminability, $d_A$',ylim=(-1.35,1.75))
ax.set_title('Analysis-specification sensitivity',loc='left',fontsize=8,weight='bold')
ax.legend(loc='lower right',fontsize=6.5,handlelength=1,ncol=1)
ax=fig.add_subplot(grid[1,3:6]);source.panel_label(ax,'e',x=-.14,y=1.09)
wide=ablation[ablation.remove_label.isin([1,2,5,10])].pivot(index='subject',columns='remove_label',values='delta_d_A').sort_index()
mat=np.vstack([wide.values,wide.median().values])
from matplotlib.colors import TwoSlopeNorm
ax.imshow(mat,cmap='PuOr',norm=TwoSlopeNorm(vmin=-.2,vcenter=0,vmax=.85),aspect='auto')
for (i,j),v in np.ndenumerate(mat):
    ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=7,color='white' if v>.48 else '#202a35',weight='bold' if i==5 else 'normal')
ax.set(yticks=range(6),yticklabels=[*wide.index,'Median'],xticks=range(4),xticklabels=[f"{n}\nP = {stats['ablation'][str(n)]['exact_signflip_p_one_sided']:.3f}" for n in [1,2,5,10]])
ax.set_title(r'Computational ablation: $\Delta d_A$',loc='left',weight='bold',fontsize=8)
ax.set_xlabel('Units removed; random minus ranked removal',fontsize=7)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
bottom=grid[2,:].subgridspec(1,2,width_ratios=[1.72,1.0],wspace=.42)
nodes,edges,profiles=source.load_archetype_data()
source.draw_archetype_network(fig.add_subplot(bottom[0,0]),fig,nodes,edges,profiles)
gp=bottom[0,1].subgridspec(2,1,hspace=.70)
active_ax=fig.add_subplot(gp[0,0]); passive_ax=fig.add_subplot(gp[1,0])
source.draw_profiles(active_ax,passive_ax,profiles)
for profile_ax, title in [(active_ax,'Active-preferring'), (passive_ax,'Passive-preferring')]:
    if profile_ax.get_legend() is not None:
        profile_ax.get_legend().remove()
    profile_ax.set_title(title+'\nLight to dark: early, middle, late',loc='left',fontsize=7,pad=5)
save(fig,'Fig2_discriminability_revised')

stats['formula']='d_A = sqrt(2) * Phi^-1(AUC); signed; chance = 0; no absolute-value folding'
stats['aggregation']='Original session AUCs were summarized within patient before transformation. Within-patient differences were calculated before their median.'
stats['inference']='Fixed-C paired gain uses the existing one-sided patient sign test. Ablation sign flips were recalculated on d_A differences. No significant-versus-nonsignificant comparison is used as a between-model test.'
stats['source_sha256']=hashlib.sha256(input_path.read_bytes()).hexdigest()
(DATA/'discriminability_summary.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2))
