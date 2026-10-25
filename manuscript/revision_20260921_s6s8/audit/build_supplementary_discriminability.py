"""Re-express existing supplementary decoder endpoints without refitting models."""
from pathlib import Path
import hashlib
import itertools
import json
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[3]/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(__file__).resolve().parents[3]/'.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.special import ndtri, ndtr
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
FIG = OUT / 'figures'
DATA = OUT / 'source_data'
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
SPEC = ROOT / 'reports/three_paper_borrowed_analyses_20260724/tables'
F2 = ROOT / 'reports/nature_science_draft/assembled_figures_20260805/source_data/fig2_population_180mm_revised'
POP = ROOT / 'neuropixel2025/data/combined_active_passive_population_candidate_panels'
BASE = ROOT / 'neuropixel2025/data/combined_active_passive_baseline_state'
TRANSFER = ROOT / 'neuropixel2025/data/combined_active_passive_sentence_pair_transfer'
INK, BLUE, TEAL, CORAL, GOLD, GRAY, PALE = '#16312C', '#437A9B', '#147D70', '#D9644A', '#D9A01C', '#6D7773', '#AAB3AF'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'axes.titlesize':7.4,'xtick.labelsize':6.3,'ytick.labelsize':6.3,
    'legend.fontsize':6.1,'axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6,
    'xtick.major.size':2.5,'ytick.major.size':2.5,'svg.fonttype':'none','pdf.fonttype':42,
    'savefig.facecolor':'white','figure.facecolor':'white','mathtext.default':'regular'})
SOURCES = {}
CONVERSIONS = []

def read(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)

def transform(x):
    a = np.asarray(x, dtype=float)
    assert np.all(np.isfinite(a)) and np.all((a > 0) & (a < 1))
    y = np.sqrt(2) * ndtri(a)
    assert np.allclose(ndtr(y / np.sqrt(2)), a, rtol=0, atol=1e-14)
    CONVERSIONS.append({'n':int(a.size),'auc_min':float(a.min()),'auc_max':float(a.max()),
                        'd_A_min':float(y.min()),'d_A_max':float(y.max())})
    return y

def signflip(diff, alternative):
    d = np.asarray(diff, dtype=float)
    null = np.array(list(itertools.product([-1,1], repeat=len(d)))) @ d / len(d)
    observed = d.mean()
    p = np.mean(null >= observed - 1e-12) if alternative == 'greater' else np.mean(np.abs(null) >= abs(observed)-1e-12)
    return float(p)

def sign_p(values):
    d=np.asarray(values,dtype=float)
    d=d[np.isfinite(d) & (np.abs(d)>1e-12)]
    return float(binomtest(int((d>0).sum()),len(d),.5,alternative='greater').pvalue)

def style(ax, title, label, chance=None, grid=True):
    ax.set_title(title, loc='left', fontweight='bold', color=INK, pad=6)
    ax.text(-.13, 1.085, label, transform=ax.transAxes, va='top', weight='bold', size=9, color=INK)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color(PALE)
    ax.tick_params(color=PALE, labelcolor=INK)
    ax.set_axisbelow(True)
    if grid: ax.grid(axis='y', color='#E7ECE9', linewidth=.55)
    if chance is not None: ax.axhline(chance, color=INK, ls=(0,(3,2)), lw=.7, zorder=1)

def medianbar(ax, x, values, width=.20):
    m = float(np.median(values)); ax.plot([x-width,x+width],[m,m],color=INK,lw=2,zorder=5)

def points(ax, arrays, xpos=None):
    if xpos is None: xpos=np.arange(len(arrays))
    for row in np.array(arrays).T:
        ax.plot(xpos,row,color='#B6C0BC',lw=.65,zorder=2)
        ax.scatter(xpos,row,s=17,color=BLUE,edgecolor='white',linewidth=.35,zorder=3)
    for x,vals in zip(xpos,arrays): medianbar(ax,x,vals)

def save(fig, stem):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    # Check all text remains within the fixed publication canvas.
    errors=[]
    for ax in fig.axes:
        box=ax.get_tightbbox(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            errors.append(ax.get_title(loc='left'))
    assert not errors, (stem, errors)
    for ext in ['pdf','svg','png','tiff']:
        kw={'dpi':400 if ext=='png' else 600}
        if ext=='tiff': kw['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(FIG/f'{stem}.{ext}', **kw)
    fig.savefig(OUT/'audit'/f'{stem}_preview.png', dpi=180)
    plt.close(fig)

def build_s6():
    spec=read(SPEC/'specification_curve.csv').sort_values('curve_order')
    sp=read(SPEC/'specification_curve_patient_values.csv')
    primary=read(F2/'panels_a_b_patient_medians.csv')
    held=read(F2/'panel_b_sentence_pair_transfer_patient_values.csv')
    null=read(TRANSFER/'sentence_pair_transfer_group_permutation_null.csv')
    original_inf=read(F2/'panel_b_sentence_pair_transfer_inference.csv')
    for c in ['patient_median_auc','bootstrap_ci_low','bootstrap_ci_high']:
        spec[c.replace('auc','d_A') if 'auc' in c else c+'_d_A']=transform(spec[c])
    sp['patient_d_A']=transform(sp.patient_auc)
    for c in ['nested_single_unit_auc','fixed_top5_auc']:
        primary[c.replace('auc','d_A')]=transform(primary[c])
    held['d_A']=transform(held.auc)
    null['group_patient_median_d_A']=transform(null.group_patient_median_auc)
    checks=[]
    for sid,g in sp.groupby('spec_id'):
        row=spec.loc[spec.spec_id==sid].iloc[0]
        assert np.isclose(g.patient_d_A.median(),row.patient_median_d_A)
        p=sign_p(g.patient_d_A)
        assert np.isclose(p,row.exact_sign_p_greater)
    for model,g in held.groupby('model'):
        obs=float(g.d_A.median());n=null.loc[null.model==model,'group_patient_median_d_A'].to_numpy()
        p=float((1+(n>=obs-1e-12).sum())/(len(n)+1))
        original=original_inf.loc[original_inf.comparison==model].iloc[0]
        assert np.isclose(p,original.group_permutation_p_greater)
        ps=sign_p(g.d_A)
        assert np.isclose(ps,original.exact_sign_p_greater)
        checks.append({'model':model,'median_d_A':obs,'sign_p':float(ps),'permutation_p':p,'n_permutations':len(n)})
    for frame,name in [(spec,'S6a_specifications'),(sp,'S6a_participant_endpoints'),(primary,'S6b_participant_endpoints'),(held,'S6c_participant_endpoints'),(null,'S6c_transformed_permutation_null')]:
        frame.to_csv(DATA/f'{name}.csv',index=False)
    fig=plt.figure(figsize=(180/25.4,130/25.4))
    a=fig.add_axes([.095,.22,.425,.69]);b=fig.add_axes([.675,.59,.30,.32]);c=fig.add_axes([.675,.14,.30,.32])
    colors={('all_trials','all_clusters'):PALE,('all_trials','good_only'):BLUE,
            ('artifact_clean_le_0p2','all_clusters'):CORAL,('artifact_clean_le_0p2','good_only'):TEAL}
    for _,r in spec.iterrows():
        isprimary=bool(r.is_primary_fixed);co=GOLD if isprimary else colors[(r.analysis_mode,r.unit_set)]
        a.vlines(r.curve_order,r.bootstrap_ci_low_d_A,r.bootstrap_ci_high_d_A,color=co,lw=.75,alpha=.8)
        a.scatter(r.curve_order,r.patient_median_d_A,s=55 if isprimary else 16,marker='*' if isprimary else 'o',color=co,zorder=3,edgecolor='white' if isprimary else co,linewidth=.4)
    a.set(xlim=(.2,32.8),ylim=(-2.9,1.75),xticks=[1,8,16,24,32],yticks=[-2,-1,0,1],
          xlabel='Specifications ordered by participant median',ylabel=r'Discriminability ($d_A$; bootstrap 95% CI)')
    style(a,'32-specification sensitivity curve','a',0)
    pr=spec.loc[spec.is_primary_fixed].iloc[0]
    a.annotate(f'Prespecified fixed top-5\nmedian {pr.patient_median_d_A:.2f}',xy=(pr.curve_order,pr.patient_median_d_A),xytext=(20,1.36),ha='left',fontsize=5.8,color='#8A6515',arrowprops={'arrowstyle':'-','color':GOLD,'lw':.7})
    handles=[Line2D([],[],color=co,marker='*' if i==4 else 'o',lw=0,ms=4,label=lab) for i,(co,lab) in enumerate([(PALE,'All trials, all clusters'),(BLUE,'All trials, good only'),(CORAL,'Artifact-clean, all clusters'),(TEAL,'Artifact-clean, good only'),(GOLD,'Prespecified fixed top-5')])]
    a.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.04,-.155),ncol=2,frameon=False,columnspacing=.7,handletextpad=.3,fontsize=5.4)
    a.text(.02,.025,f'Range {spec.patient_median_d_A.min():.2f} to {spec.patient_median_d_A.max():.2f}; 27/32 medians > 0',transform=a.transAxes,fontsize=5.8,color=GRAY)
    points(b,[primary.nested_single_unit_d_A,primary.fixed_top5_d_A])
    b.set(xticks=[0,1],xticklabels=['Nested\nsingle unit','Fixed\ntop-5'],ylim=(-.9,1.5),yticks=[-.5,0,.5,1],ylabel=r'Discriminability ($d_A$)')
    style(b,'Decoder-specification endpoints','b',0)
    b.text(.98,.04,f'Top-5 median {primary.fixed_top5_d_A.median():.2f}\nSign P = 0.1875',transform=b.transAxes,ha='right',va='bottom',fontsize=5.7,color=TEAL)
    pivot=held.pivot(index='subject',columns='model',values='d_A')
    points(c,[pivot.nested_single_unit,pivot.fixed_top5_population])
    c.set(xticks=[0,1],xticklabels=['Nested\nsingle unit','Fixed\ntop-5'],ylim=(-1.1,1.8),yticks=[-.5,0,.5,1,1.5],ylabel=r'Held-pair discriminability ($d_A$)')
    style(c,'Held-sentence-pair transfer','c',0)
    c.text(.98,.03,'Top-5 median 0.58\nPooled permutation P = 0.024\nParticipant sign P = 0.1875',transform=c.transAxes,ha='right',va='bottom',fontsize=5.6,color=TEAL)
    save(fig,'FigS6_decoder_specification_discriminability')
    return {'range_d_A':[float(spec.patient_median_d_A.min()),float(spec.patient_median_d_A.max())],
            'primary_median_d_A':float(pr.patient_median_d_A),'primary_ci_d_A':[float(pr.bootstrap_ci_low_d_A),float(pr.bootstrap_ci_high_d_A)],
            'n_specifications':32,'n_participants':5,'positive_medians':27,'inference_checks':checks,'bootstrap_ci_limits_transformed':True}

def build_s7():
    ab=read(POP/'ablation_patient_values.csv');sh=read(POP/'trial_shuffle_patient_values.csv')
    dist=read(POP/'cv_population_distance_patient_values.csv');oldab=read(POP/'ablation_group_summary.csv');oldsh=read(POP/'trial_shuffle_group_summary.csv')
    for frame,cols in [(ab,['ranked_removal_auc','random_removal_median_auc']),(sh,['simultaneous_auc','trial_shuffled_median_auc'])]:
        for col in cols: frame[col.replace('auc','d_A')]=transform(frame[col])
    ab['random_minus_ranked_d_A']=ab.random_removal_median_d_A-ab.ranked_removal_d_A
    sh['simultaneous_minus_shuffled_d_A']=sh.simultaneous_d_A-sh.trial_shuffled_median_d_A
    abrows=[]
    for k,g in ab.groupby('remove_order'):
        p=signflip(g.random_minus_ranked_d_A,'greater');original=oldab.loc[oldab.remove_order==k].iloc[0]
        assert np.isclose(p,original.random_minus_ranked_exact_sign_flip_p_greater)
        r={'remove_order':k,'remove_label':str(g.remove_label.iloc[0]),'n_participants':len(g),'median_random_minus_ranked_d_A':g.random_minus_ranked_d_A.median(),'exact_sign_flip_p_greater':p}
        for prefix,col in [('ranked','ranked_removal_d_A'),('random','random_removal_median_d_A')]:
            for suffix,q in [('q25',.25),('median',.5),('q75',.75)]:r[f'{prefix}_{suffix}_d_A']=g[col].quantile(q)
        abrows.append(r)
    ag=pd.DataFrame(abrows);shrows=[]
    for m,g in sh.groupby('model'):
        p=signflip(g.simultaneous_minus_shuffled_d_A,'two-sided');original=oldsh.loc[oldsh.model==m].iloc[0]
        assert np.isclose(p,original.simultaneous_minus_shuffled_exact_sign_flip_p_two_sided)
        shrows.append({'model':m,'n_participants':len(g),'median_difference_d_A':float(g.simultaneous_minus_shuffled_d_A.median()),'exact_sign_flip_p_two_sided':p})
    for frame,name in [(ab,'S7a_participant_endpoints'),(ag,'S7a_group_summary'),(sh,'S7b_participant_endpoints'),(pd.DataFrame(shrows),'S7b_group_summary'),(dist,'S7c_population_distance_unchanged')]:frame.to_csv(DATA/f'{name}.csv',index=False)
    fig=plt.figure(figsize=(180/25.4,130/25.4))
    a=fig.add_axes([.095,.17,.425,.74]);b=fig.add_axes([.675,.59,.30,.32]);c=fig.add_axes([.675,.14,.30,.32])
    for prefix,co,lab in [('ranked',CORAL,'Remove train-ranked units'),('random',GRAY,'Remove random units')]:
        a.fill_between(ag.remove_order,ag[f'{prefix}_q25_d_A'],ag[f'{prefix}_q75_d_A'],color=co,alpha=.14,lw=0)
        a.plot(ag.remove_order,ag[f'{prefix}_median_d_A'],'o-',color=co,lw=1.2,ms=3.7,label=lab)
    a.set(xticks=ag.remove_order,xticklabels=ag.remove_label,ylim=(-1.1,.6),yticks=[-1,-.5,0,.5],xlabel='Units removed within each training fold',ylabel=r'Discriminability ($d_A$; participant median and IQR)')
    style(a,'Ranked versus random unit ablation','a',0)
    a.legend(loc='upper left',bbox_to_anchor=(-.02,-.10),frameon=False,fontsize=6)
    a.text(.98,.045,'Ranked-removal disadvantage\nExact P = 0.03125 at 1, 2 and 5 units',transform=a.transAxes,ha='right',fontsize=5.8,color=CORAL)
    for model,xs in [('all',[0,1]),('top5',[2.4,3.4])]:
        g=sh.loc[sh.model==model];points(b,[g.simultaneous_d_A,g.trial_shuffled_median_d_A],xs)
    b.set(xticks=[0,1,2.4,3.4],xticklabels=['Simult.\nall','Shuffled\nall','Simult.\ntop-5','Shuffled\ntop-5'],ylim=(-.95,1.5),yticks=[-.5,0,.5,1],ylabel=r'Discriminability ($d_A$)')
    style(b,'Condition-preserving trial shuffle','b',0)
    delta=next(x for x in shrows if x['model']=='top5')['median_difference_d_A']
    b.text(.98,.04,f'Top-5 median $\Delta d_A$ = {delta:.3f}\nTwo-sided exact P = 1.0',transform=b.transAxes,ha='right',fontsize=5.7,color=TEAL)
    rng=np.random.default_rng(20260809)
    for x,m in enumerate(['all','top5']):
        vals=dist.loc[dist.model==m,'cv_population_distance'].to_numpy();jitter=rng.uniform(-.07,.07,len(vals))
        c.scatter(x+jitter,vals,s=21,color=BLUE,edgecolor='white',lw=.4,zorder=3);medianbar(c,x,vals)
    c.set(xticks=[0,1],xticklabels=['All units','Top-5'],ylim=(-.28,.57),yticks=[-.2,0,.2,.4],ylabel='Cross-validated distance')
    style(c,'Cross-validated population distance','c',0)
    c.text(.98,.04,'Top-5 median 0.148\nOne-sided exact P = 0.0625',transform=c.transAxes,ha='right',fontsize=5.7,color=TEAL)
    save(fig,'FigS7_population_discriminability_diagnostics')
    return {'ablation':abrows,'shuffle':shrows,'n_participants':5,'panel_c_metric_unchanged':True,'sign_flip_tests_recomputed_on_d_A':True,'all_recomputed_p_values_match_original':True}

def build_s8():
    df=read(BASE/'active_passive_baseline_state_run_summary.csv');corr=read(BASE/'active_passive_baseline_state_unit_correlations.csv')
    cols=['baseline_auc','raw_response_auc','response_auc','residual_response_auc']
    for col in cols:df[col.replace('auc','d_A')]=transform(df[col])
    df.to_csv(DATA/'S8_session_endpoints.csv',index=False);corr.to_csv(DATA/'S8d_unit_correlations_unchanged.csv',index=False)
    fig=plt.figure(figsize=(180/25.4,130/25.4))
    a=fig.add_axes([.095,.595,.36,.315]);b=fig.add_axes([.615,.595,.36,.315]);c=fig.add_axes([.095,.145,.36,.315]);d=fig.add_axes([.615,.145,.36,.315])
    def unpaired(ax,cols,labels):
        for x,col in enumerate(cols):
            vals=df[col].to_numpy();jitter=np.linspace(-.18,.18,len(vals))
            ax.scatter(x+jitter,vals,s=16,color=BLUE,alpha=.84,zorder=3);medianbar(ax,x,vals,.26)
        ax.set_xticks(np.arange(len(cols)),labels)
    unpaired(a,[x.replace('auc','d_A') for x in cols],['Baseline','Raw\nevoked','Baseline-\ncorrected','Corrected\nresidual'])
    a.set(ylim=(-1.3,1.8),yticks=[-1,0,1],ylabel=r'Held-pair discriminability ($d_A$)')
    style(a,'Baseline, evoked and residual decoding','a',0)
    unpaired(b,['baseline_cv_axis_frac_pair_positive','response_cv_axis_frac_pair_positive','residual_response_cv_axis_frac_pair_positive'],['Baseline','Evoked','Evoked\nresidual'])
    b.set(ylim=(.35,1),yticks=[.4,.6,.8,1],ylabel='Fraction of positive held-out pairs')
    style(b,'Held-pair axis projection','b',.5)
    unpaired(c,['baseline_response_axis_cosine','baseline_residual_axis_cosine'],['Baseline versus\nevoked','Baseline versus\nresidual'])
    c.set(ylim=(-.7,.5),yticks=[-.6,-.3,0,.3],ylabel='Cosine similarity')
    style(c,'Baseline and evoked axis alignment','c',0)
    values=corr.baseline_response_r.dropna().to_numpy();counts,bins=np.histogram(values,bins=35)
    d.hist(values,bins=bins,color=BLUE,alpha=.82);d.axvline(0,color=GRAY,lw=.8,ls='--')
    d.set(xlim=(-1.02,1.02),xticks=[-1,-.5,0,.5,1],xlabel='Within-unit baseline-response correlation',ylabel='Unit count')
    style(d,'Baseline-response correlations','d',grid=False)
    pd.DataFrame({'bin_left':bins[:-1],'bin_right':bins[1:],'count':counts}).to_csv(DATA/'S8d_histogram_unchanged.csv',index=False)
    save(fig,'FigS8_baseline_state_discriminability')
    return {'n_sessions':len(df),'n_participants':int(df.subject.nunique()),'median_d_A':{col.replace('auc','d_A'):float(df[col.replace('auc','d_A')].median()) for col in cols},'panels_b_c_d_metrics_and_source_values_unchanged':True,'correlations_displayed':len(values),'new_significance_test':False}

def main():
    result={'figure_contract':{'conclusion':'Existing decoder specification, ablation, covariance and baseline controls expressed on a common signed discriminability scale.',
        'evidence':'Previously saved participant or session endpoints; model fitting, exclusions and validation splits preserved.',
        'archetype':'Quantitative grids retaining the existing three-panel and four-panel evidence structure.',
        'backend':'Python matplotlib','exports':'180 x 130 mm editable PDF/SVG, 400 dpi PNG, 600 dpi TIFF and source CSVs.',
        'statistical_boundary':'Re-expression of effect sizes does not create independent evidence; exploratory tests retain their original comparison and sidedness.'},
        'formula':'d_A = sqrt(2) * Phi_inverse(AUC)','chance':0,'negative_values_retained':True,
        'S6':build_s6(),'S7':build_s7(),'S8':build_s8(),'source_sha256':SOURCES,
        'all_transformations_finite':True,'all_inverse_transformations_match_original':True,'conversion_batches':CONVERSIONS}
    (OUT/'audit/figure_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['S6','S7','S8']},indent=2))

if __name__=='__main__':main()
