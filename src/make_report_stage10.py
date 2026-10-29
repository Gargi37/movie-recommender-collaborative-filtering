"""Builds docs/evaluation_report.md and reports/figures/s10_*.png from results/test_bundle.pkl (no test labels are read here),
and appends the test rows to the experiment table."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.experiments import log

sns.set_theme(style='whitegrid')
B = pd.read_pickle('results/test_bundle.pkl'); V = pd.read_pickle('results/val_bundle.pkl')
res, noo, sig, div, ls, fair, gaps, err, calib, bias_tbl, expl, choice = [B[k] for k in ['res_all', 'noo' if False else 'res_noorig', 'sig', 'div', 'lang_share', 'fair', 'gaps', 'err', 'calib', 'bias_tbl', 'expl', 'choice']]
grid = pd.read_csv('results/mitigation_grid_val.csv')
FIG = Path('reports/figures')
DEP, SAFE, DIV = 'Item CF + popularity (deployed)', 'Deployed + MMR/exposure (safe)', 'Deployed + MMR/exposure (diverse)'
r = res[res.task == 'ranking'].set_index('model'); rt = res[res.task == 'rating'].set_index('model')   # ranking / rating lookups (Item CF, SVD appear in both)
def S(a, b, met):
    row = sig[(sig.comparison == f'{a} - {b}') & sig.metric.str.startswith(met)].iloc[0]; return row.estimate, row.ci_low, row.ci_high
md = lambda d, f='.3f': d.to_markdown(index=False, floatfmt=f)

# ---------------------------------------------------------------- figures
def forest(rows, title, fname, xlabel):
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.5))
    for i, (lab, e, lo, hi) in enumerate(rows):
        ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt='o', color='#4C78A8', capsize=4)
    ax.axvline(0, color='k', ls='--'); ax.set_yticks(range(len(rows))); ax.set_yticklabels([x[0] for x in rows]); ax.invert_yaxis()
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel('Comparison'); plt.tight_layout(); plt.savefig(FIG / fname, dpi=150); plt.close()
forest([(c.replace(' - ', '  minus  '), e, lo, hi) for c, e, lo, hi in sig[sig.metric.str.startswith('RMSE')][['comparison', 'estimate', 'ci_low', 'ci_high']].itertuples(index=False)],
       'Test set: RMSE differences with 95% bootstrap CI (left of 0 = first model better)', 's10_01_rmse_differences.png', 'RMSE difference')
forest([(c.replace(' - ', '  minus  ').replace('Item CF + popularity (deployed)', 'Deployed blend').replace('Deployed + MMR/exposure', 'Re-ranked'), e, lo, hi)
        for c, e, lo, hi in sig[sig.metric.str.startswith('P@10')][['comparison', 'estimate', 'ci_low', 'ci_high']].itertuples(index=False)],
       'Test set: P@10 differences with 95% bootstrap CI (right of 0 = first model better)', 's10_02_p10_differences.png', 'P@10 difference')

fig, ax = plt.subplots(figsize=(8, 5.5))
sc = ax.scatter(grid.coverage, grid.P10, c=grid.lam, cmap='viridis', s=40 + 25 * grid.gamma)
plt.colorbar(sc, ax=ax, label='MMR weight lambda (1 = none)')
ax.text(0.02, 0.03, 'Marker size = exposure penalty gamma.\nCoverage is capped near 0.5 because only each\nuser\'s top-50 candidates are re-ranked.', transform=ax.transAxes, fontsize=8, va='bottom')
for k, lab, col in [('safe', 'safe', 'red'), ('diverse', 'diverse', 'orange')]:
    p = choice[k]; row = grid[(grid.lam == p['lam']) & (grid.gamma == p['gamma'])].iloc[0]
    ax.scatter(row.coverage, row.P10, marker='*', s=300, color=col, edgecolor='k', label=f'{lab} (chosen on validation)')
ax.set_title('Validation: accuracy vs catalogue coverage for re-ranker settings'); ax.set_xlabel('Coverage@10 (share of catalogue recommended)'); ax.set_ylabel('P@10'); ax.legend(loc='upper right')
plt.tight_layout(); plt.savefig(FIG / 's10_03_accuracy_diversity_tradeoff.png', dpi=150); plt.close()

cols = ['Catalogue', 'Relevant test ratings', 'Popularity', 'Item CF', DEP, SAFE]
ls[cols].mul(100).plot.bar(figsize=(11, 4.8), width=0.85); plt.title('Language share of recommendations vs catalogue and relevant ratings (test)')
plt.xlabel('Language'); plt.ylabel('Share (%)'); plt.xticks(rotation=0); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG / 's10_04_language_share.png', dpi=150); plt.close()

order = ['Random', 'Popularity', 'Item CF', 'SVD', DEP, SAFE, DIV]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
for a, col, t in zip(ax, ['coverage', 'tail_share_of_recs', 'gini_exposure'], ['Catalogue coverage@10', 'Share of recommendations from tail movies', 'Gini of exposure (0 = equal)']):
    a.barh(order, div.loc[order, col], color='#4C78A8'); a.set_title(t); a.set_xlabel(t); a.invert_yaxis()
ax[0].set_ylabel('Model'); plt.tight_layout(); plt.savefig(FIG / 's10_05_exposure.png', dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
e1 = err[err.by == 'rating_bucket']; x = np.arange(len(e1))
for j, (c, lab) in enumerate([('RMSE_global_mean', 'Global mean'), ('RMSE_bias', 'Bias baseline'), ('RMSE_item_cf', 'Item CF')]):
    ax[0].bar(x + (j - 1) * .27, e1[c], .27, label=lab)
ax[0].set_xticks(x); ax[0].set_xticklabels(e1.group); ax[0].set_title('RMSE by actual rating (models pull ratings to the middle)'); ax[0].set_xlabel('Actual rating (stars)'); ax[0].set_ylabel('RMSE'); ax[0].legend()
e2 = err[err.by == 'language']; x = np.arange(len(e2))
for j, (c, lab) in enumerate([('RMSE_global_mean', 'Global mean'), ('RMSE_item_cf', 'Item CF')]):
    ax[1].bar(x + (j - .5) * .38, e2[c], .38, label=lab)
ax[1].set_xticks(x); ax[1].set_xticklabels(e2.group, rotation=20); ax[1].set_title('RMSE by movie language'); ax[1].set_xlabel('Language'); ax[1].set_ylabel('RMSE'); ax[1].legend()
plt.tight_layout(); plt.savefig(FIG / 's10_06_error_analysis.png', dpi=150); plt.close()

fg = fair.copy(); fg['label'] = fg.grouping.str.split(' ').str[0] + ': ' + fg.group.astype(str)
fig, ax = plt.subplots(figsize=(11, 4.8)); x = np.arange(len(fg))
ax.bar(x - .2, fg['RMSE Item CF'], .4, label='Item CF'); ax.bar(x + .2, fg['RMSE Bias baseline'], .4, label='Bias baseline')
ax.set_xticks(x); ax.set_xticklabels(fg.label, rotation=40, ha='right'); ax.set_title(f'Test RMSE by user group ({int(fair.users.min())}-{int(fair.users.max())} users per group)'); ax.set_xlabel('User group'); ax.set_ylabel('RMSE'); ax.legend()
plt.tight_layout(); plt.savefig(FIG / 's10_07_fairness_groups.png', dpi=150); plt.close()

# ---------------------------------------------------------------- criteria
rm = lambda m: rt.loc[m, 'RMSE']
e1, l1, h1 = S('Item CF', 'Item mean', 'RMSE'); e2, l2, h2 = S('Item CF', 'Global mean', 'RMSE'); e3, l3, h3 = S('Item CF', 'Bias baseline', 'RMSE')
ep, lp, hp = S(DEP, 'Popularity', 'P@10'); en, ln, hn = S(DEP, 'Popularity', 'NDCG@10')
ild = lambda m: (div.loc[m, 'ild_genre'] + div.loc[m, 'ild_language']) / 2
loss = 1 - r.loc[SAFE, 'P@10'] / r.loc[DEP, 'P@10']
act = gaps.loc[('activity (train ratings)', 'Item CF')]
crit = pd.DataFrame([
    ['1', 'Best CF RMSE below item-mean baseline', 'Lower', f'{rm("Item CF"):.4f} vs {rm("Item mean"):.4f} (diff {e1:+.4f}, CI [{l1:+.4f}, {h1:+.4f}])', 'Met' if h1 < 0 else 'Not met'],
    ['2', 'Best CF RMSE below global-mean baseline', 'Lower', f'{rm("Item CF"):.4f} vs {rm("Global mean"):.4f} (diff {e2:+.4f}, CI [{l2:+.4f}, {h2:+.4f}])', 'Met' if h2 < 0 else 'Not met'],
    ['3', 'P@10 above popularity baseline', 'Higher', f'{r.loc[DEP, "P@10"]:.4f} vs {r.loc["Popularity", "P@10"]:.4f} (diff {ep:+.4f}, CI [{lp:+.4f}, {hp:+.4f}])', 'Met' if lp > 0 else 'Not met (statistical tie)'],
    ['4', 'Catalogue coverage@10', '>= 30% (assumed target)', f'deployed {r.loc[DEP, "Coverage@10"]:.1%}; with safe re-ranker {r.loc[SAFE, "Coverage@10"]:.1%}; diverse {r.loc[DIV, "Coverage@10"]:.1%}', 'Not met by deployed model; met with re-ranker'],
    ['5', 'Re-ranker raises diversity with <= 5% relative P@10 loss', 'Improves, loss <= 5%', f'combined ILD {ild(DEP):.3f} -> {ild(SAFE):.3f}; P@10 {r.loc[DEP, "P@10"]:.4f} -> {r.loc[SAFE, "P@10"]:.4f} ({-loss:+.1%})', 'Met' if (ild(SAFE) > ild(DEP) and loss <= .05) else 'Not met'],
    ['6', 'RMSE gap across user-activity groups', '<= 0.15', f'{act.iloc[0]:.3f} (bootstrap 95% range {act.iloc[1]:.3f}-{act.iloc[2]:.3f}; max-min is biased upward)', 'Met' if act.iloc[0] <= .15 else 'Not met'],
    ['7', 'API and UI', 'All endpoint tests pass', 'Built in Stage 11', 'Pending'],
], columns=['#', 'Criterion (Stage 1)', 'Target', 'Result on test set', 'Met?'])
crit.to_csv('results/stage1_criteria_results.csv', index=False)

# ---------------------------------------------------------------- report text
Vs = V['sig']; Vr = V['res_all'][V['res_all'].task == 'ranking'].set_index('model'); Vt = V['res_all'][V['res_all'].task == 'rating'].set_index('model')
_v = Vs[(Vs.comparison == f'{DEP} - Popularity') & Vs.metric.str.startswith('P@10')].iloc[0]; vp, vl, vh = _v.estimate, _v.ci_low, _v.ci_high
def vloss(k):
    p_ = choice[k]; row = grid[(grid.lam == p_['lam']) & (grid.gamma == p_['gamma'])].iloc[0]; return row.P10_rel_loss
tloss = lambda m: 1 - r.loc[m, 'P@10'] / r.loc[DEP, 'P@10']
reg = ['Tamil', 'Telugu', 'Kannada', 'Malayalam']
rs = lambda c: ls.loc[reg, c].sum() * 100
tg = fair[fair.grouping == 'generator taste group']
ab = err[err.by == 'rating_bucket'].set_index('group')
orig = err[err.by == 'is_original'].set_index('group')
rate_t = res[res.task == 'rating'][['model', 'RMSE', 'MAE']]
rate_t['RMSE excl. original rows'] = noo[noo.task == 'rating'].RMSE.to_numpy()
rank_t = res[res.task == 'ranking'][['model', 'P@5', 'P@10', 'R@10', 'NDCG@10', 'Hit@10', 'Coverage@10']].copy()
rank_t['P@10 excl. original rows'] = noo[noo.task == 'ranking']['P@10'].to_numpy()
div_t = div[['ild_genre', 'ild_language', 'coverage', 'gini_exposure', 'tail_share_of_recs', 'tail_items_recommended', 'novelty_bits', 'pop_exposure_spearman', 'unexpected_share', 'serendipity_per_list', 'taste_alignment']].reset_index().rename(columns={'index': 'model'})
ab_txt = f"movies actually rated 1-2 stars are predicted {ab.loc['1-2', 'mean_error']:+.2f} stars too high and movies rated 4.5-5 are predicted {ab.loc['4.5-5', 'mean_error']:+.2f} too low"
L = ['# Stage 10: Evaluation, validation and audit (auto-generated by src/make_report_stage10.py)', '',
 '## 1. Protocol (fixed before the test set was opened)',
 '- Train on train_full (4,029 ratings); score the locked test set (971 ratings, 284 users with at least one relevant movie). Test SHA-256 verified; results/test_opened.txt records the single run.',
 '- Deployed model (declared in Stage 9): item-based CF + popularity blend. Re-ranker settings ("safe", "diverse") were chosen on VALIDATION by a stated rule. Nothing was reselected after seeing test results.',
 '- Relevance for ranking: rating >= 4.0; unrated movies count as non-relevant (pessimistic). Candidates exclude movies the user already rated in training.',
 '- Uncertainty: paired bootstrap over users (2,000 resamples, seed 42).',
 f"- Caveat: {err[err.by == 'is_original'].set_index('group').loc['True', 'n'] / B['n_target']:.0%} of test ratings come from the provided original file and carry no taste signal (RMSE {orig.loc['True', 'RMSE_item_cf']:.3f} vs {orig.loc['False', 'RMSE_item_cf']:.3f} on generated rows). Tables also show results without them.",
 '', '## 2. Headline results', '', '### Rating prediction', '', md(rate_t, '.4f'), '', '### Top-10 ranking', '', md(rank_t, '.4f'), '',
 '(Coverage@10 here is over all users. Random is the mean of 50 seeds.)', '',
 '## 3. Is the difference real? (paired bootstrap over users, 95% CI)', '', md(sig[['comparison', 'metric', 'estimate', 'ci_low', 'ci_high']], '.4f'), '',
 '![RMSE differences](../reports/figures/s10_01_rmse_differences.png)', '', '![P@10 differences](../reports/figures/s10_02_p10_differences.png)', '',
 '**Reading:**',
 f'- CF beats the item-mean and global-mean baselines clearly ({e1:+.3f} and {e2:+.3f} RMSE) but is indistinguishable from "global mean + user bias + movie bias" ({e3:+.4f}, CI [{l3:+.4f}, {h3:+.4f}]). Matrix factorisation is nominally best ({rm("SVD"):.4f}) but not significantly. Most of the accuracy comes from knowing which movies are generally liked and which users rate generously.',
 f'- Pure CF ranking is significantly WORSE than popularity (P@10 {S("Item CF", "Popularity", "P@10")[0]:+.4f}). The CF + popularity blend ties popularity ({ep:+.4f}, CI [{lp:+.4f}, {hp:+.4f}]) and is significantly better than pure CF ({S(DEP, "Item CF", "P@10")[0]:+.4f}).',
 f'- On validation the blend had looked clearly better than popularity (P@10 {vp:+.4f}, 95% CI [{vl:+.4f}, {vh:+.4f}]). On test that advantage disappeared. The blend weight was chosen on the same data, so the validation result was optimistic: a concrete example of selection bias, and the reason the test set was kept sealed.',
 '- Plausible reason popularity is hard to beat here: in the synthetic data users rate popular movies more often, so popularity predicts which movies get rated later, which is what the ranking test measures.',
 '', '## 4. Stage 1 success criteria', '', md(crit), '',
 '## 5. Error analysis (deployed rating model = item-based CF)', '', md(err[['by', 'group', 'n', 'RMSE_item_cf', 'RMSE_bias', 'RMSE_global_mean', 'mean_error']]), '',
 '![Error analysis](../reports/figures/s10_06_error_analysis.png)', '',
 f'- Predictions are pulled towards the middle: {ab_txt}. Typical of rating models with noisy data.',
 f"- Errors are similar across languages, genres, eras and activity levels; rare movies are harder than popular ones ({err[(err.by == 'pop_tercile') & (err.group == 'rare movies')].RMSE_item_cf.iloc[0]:.3f} vs {err[(err.by == 'pop_tercile') & (err.group == 'popular movies')].RMSE_item_cf.iloc[0]:.3f}).",
 '- Group differences of a few hundredths of RMSE are within noise at these sample sizes (about 90-340 ratings per group).',
 '', '### Calibration (predicted vs actual rating)', '', md(calib), '',
 '## 6. What drives the model (explainability)',
 f'- Ablation: global mean {rm("Global mean"):.3f} -> + movie effect {rm("Item mean"):.3f} -> + user effect {rm("Bias baseline"):.3f} -> + CF neighbours {rm("Item CF"):.3f}. Nearly all accuracy is in the biases.',
 '- Most positive and negative movie biases (deployed model, trained on train_full):', '', md(bias_tbl.reset_index().rename(columns={'index': 'movie_id'}), '.3f'), '',
 f"- SVD item factors vs genre+language: R-squared {expl['svd_item_factors_R2_on_genre_language']:.3f} against {expl['permutation_null_R2']:.3f} for shuffled metadata (the chance level). The latent factors do not line up with genre or language, so they cannot honestly be described as genre tastes.",
 '- Per-recommendation explanation: Recommender.explain(user, movie) returns the global mean, user bias, movie bias, CF adjustment and the rated movies that pulled the prediction most (used by the API in Stage 11).',
 '', '## 7. Business-logic checks',
 f"- Hit@10 for the deployed model is {r.loc[DEP, 'Hit@10']:.1%} (a user gets at least one later-liked movie in the top 10) versus {r.loc['Random', 'Hit@10']:.1%} for random.",
 '- Recommended lists never contain movies the user already rated (asserted in tests/test_audit.py and tests/test_models.py).',
 '- Unknown users receive the popularity fallback; invalid k raises an error (tests/test_models.py).',
 f"- Hidden-taste check (audit only): average alignment of recommended movies with each user's generator taste group is {div.loc[DEP, 'taste_alignment']:.3f} for the deployed model, {div.loc['Item CF', 'taste_alignment']:.3f} for pure CF, {div.loc['Popularity', 'taste_alignment']:.3f} for popularity and {div.loc['Random', 'taste_alignment']:.3f} for random. **No model beats random**, so there is no evidence that the recommender recovers the built-in taste groups from this little data.",
 '', '## 8. Diversity, popularity bias and fairness audit (mandatory)', '', md(div_t), '',
 '![Exposure](../reports/figures/s10_05_exposure.png)', '',
 f"- **Popularity bias / filter bubble:** the deployed blend recommends only {int(div.loc[DEP, 'tail_items_recommended'])} of 101 tail movies (tail share {div.loc[DEP, 'tail_share_of_recs']:.1%}); catalogue coverage is {div.loc[DEP, 'coverage']:.1%} and exposure Gini {div.loc[DEP, 'gini_exposure']:.2f}. Everyone gets almost the same list. The popularity-heavy models (popularity itself and the blend) are the ones that score highest on P@10, while pure CF scores lower but is more varied, so accuracy and diversity pull in opposite directions.",
 f"- **Regional-language exposure:** Tamil, Telugu, Kannada and Malayalam are {rs('Catalogue'):.1f}% of the catalogue and {rs('Relevant test ratings'):.1f}% of relevant test ratings; the deployed blend gives them {rs(DEP):.1f}% of slots and the safe re-ranker {rs(SAFE):.1f}%. Within that, Telugu is over-exposed ({ls.loc['Telugu', DEP] * 100:.1f}% vs {ls.loc['Telugu', 'Catalogue'] * 100:.1f}% of the catalogue) and English under-exposed ({ls.loc['English', DEP] * 100:.1f}% vs {ls.loc['English', 'Catalogue'] * 100:.1f}%).",
 '', md(ls.mul(100).reset_index().rename(columns={'index': 'language'}), '.1f'), '', '![Language share](../reports/figures/s10_04_language_share.png)', '',
 '### Mitigation: MMR re-ranking with an exposure penalty',
 f"- Rule (set on validation): maximise coverage@10 subject to relative P@10 loss <= 5% (safe) or <= 20% (diverse). Chosen: safe = lambda {choice['safe']['lam']}, gamma {choice['safe']['gamma']}; diverse = lambda {choice['diverse']['lam']}, gamma {choice['diverse']['gamma']}.",
 f"- Test: safe raises coverage {r.loc[DEP, 'Coverage@10']:.1%} -> {r.loc[SAFE, 'Coverage@10']:.1%}, combined list diversity {ild(DEP):.3f} -> {ild(SAFE):.3f}, tail share {div.loc[DEP, 'tail_share_of_recs']:.1%} -> {div.loc[SAFE, 'tail_share_of_recs']:.1%}, at P@10 {r.loc[DEP, 'P@10']:.4f} -> {r.loc[SAFE, 'P@10']:.4f} (CI of the difference includes 0). Diverse reaches coverage {r.loc[DIV, 'Coverage@10']:.1%} at P@10 {r.loc[DIV, 'P@10']:.4f}.",
 f'- Honest limits: coverage cannot exceed about 0.5 because only the top-50 candidates of each user are re-ranked. Relative P@10 loss was {vloss("safe"):.1%} (safe) and {vloss("diverse"):.1%} (diverse) on validation versus {tloss(SAFE):.1%} and {tloss(DIV):.1%} on test; small samples make both noisy. The exposure penalty depends on user processing order; a live system would use running counts.',
 '', '![Trade-off](../reports/figures/s10_03_accuracy_diversity_tradeoff.png)', '',
 '### Group fairness (no demographic data exists: groups are activity, rating generosity and the hidden generator taste group)', '', md(fair.drop(columns=[c for c in fair.columns if 'SVD' in c]), '.3f'), '', md(gaps.reset_index().rename(columns={'level_0': 'grouping', 'level_1': 'model'}), '.3f'), '',
 '![Fairness](../reports/figures/s10_07_fairness_groups.png)', '',
 f"- Activity-level gap {act.iloc[0]:.3f} is under the 0.15 criterion. The taste-group gap ({gaps.loc[('generator taste group', 'Item CF')].iloc[0]:.3f}) is about the same for the bias baseline ({gaps.loc[('generator taste group', 'Bias baseline')].iloc[0]:.3f}), so it reflects group noise and sample size ({int(tg.users.min())}-{int(tg.users.max())} users per group), not something CF introduces. The bootstrap range of a max-min gap is biased upward.",
 '- Demographic fairness cannot be assessed: the dataset has no demographic attributes, by design.',
 '', '## 9. Limitations (state these in the report)',
 '- Synthetic data: effects were built into the generator, ratings are not real viewing behaviour, and results may not transfer.',
 '- Sparse data (about 13 ratings per user): personalisation adds almost nothing beyond biases; the learning curve showed no growth with more data.',
 '- Popularity confound: the ranking test rewards predicting what users will rate, not only what they will like.',
 '- About 13% of test ratings come from the original noisy rows; test RMSE is slightly pessimistic.',
 '- Test set is small (971 ratings, 284 scored users); confidence intervals are wide. Per-user split is not a strict global-time replay.',
 '- Blend weight and re-ranker settings were chosen on validation; the validation-to-test drop above shows how optimistic such choices can be.',
 '- No real demographic or user-consent data; no online A/B test.',
 '', '## 10. What changed between validation and test',
 '| Quantity | Validation | Test |', '|---|---|---|',
 f"| Deployed blend P@10 minus popularity | {vp:+.4f} (CI [{vl:+.4f}, {vh:+.4f}]) | {ep:+.4f} (CI [{lp:+.4f}, {hp:+.4f}]) |",
 f"| Safe re-ranker coverage@10 | {Vr.loc[SAFE, 'Coverage@10']:.1%} | {r.loc[SAFE, 'Coverage@10']:.1%} |",
 f"| Item CF RMSE | {Vt.loc['Item CF', 'RMSE']:.4f} | {rm('Item CF'):.4f} |"]
Path('docs/evaluation_report.md').write_text('\n'.join(L)); print('\n'.join(L)[:1500])

# experiment table rows (test)
rows = []
for i, m in enumerate(['Global mean', 'User mean', 'Item mean', 'Bias baseline', 'Random', 'Popularity', 'Top-rated', 'Item CF', 'User CF', 'SVD', DEP, 'SVD + popularity', SAFE, DIV]):
    rr = res[res.model == m]; rk = rr[rr.task == 'ranking']; ra = rr[rr.task == 'rating']
    if m == DEP:
        ra = res[res.model == 'Item CF'].query("task == 'rating'")
    g = lambda d, c: float(d[c].iloc[0]) if len(d) else np.nan
    rows.append([f'E{16 + i}', 10, m, 'tuned (Stage 9)' if m not in ('Global mean', 'User mean', 'Item mean', 'Random', 'Popularity', 'Top-rated') else '-', 'train_full', 'test',
                 g(ra, 'RMSE'), g(ra, 'MAE'), g(rk, 'P@10'), g(rk, 'R@10'), g(rk, 'Hit@10'), g(rk, 'Coverage@10'), g(rk, 'users_scored'), 'FINAL test evaluation (opened once)'])
log(rows)
