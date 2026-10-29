"""Collects every number/table the Word report needs from saved result files -> reports/report_data.json (no numbers are typed by hand)."""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

R = Path('results'); D = Path('docs'); F = Path('reports/figures')
read = lambda p: pd.read_csv(p)
f3 = lambda v: '' if pd.isna(v) else f'{v:.3f}'
f4 = lambda v: '' if pd.isna(v) else f'{v:.4f}'
pc = lambda v: f'{v * 100:.1f}%'

ds = pd.read_csv('data/processed/dataset.csv'); cat = pd.read_csv('data/raw/movie_catalogue.csv'); raw = pd.read_csv('data/raw/dataset.csv')
pu, pi = ds.groupby('user_id').size(), ds.groupby('movie_id').size()
tc, va, tf, te = [pd.read_csv(f'data/processed/split/{n}.csv') for n in ['train_core', 'val', 'train_full', 'test']]
tm, tmn, sig = read(R / 'test_metrics.csv'), read(R / 'test_metrics_no_original.csv'), read(R / 'test_significance.csv')
rate, rank = tm[tm.task == 'rating'].set_index('model'), tm[tm.task == 'ranking'].set_index('model')
div = read(R / 'test_diversity_audit.csv').rename(columns={'Unnamed: 0': 'model'}).set_index('model')
lang = read(R / 'test_language_share.csv').rename(columns={'Unnamed: 0': 'language'}).set_index('language')
fair, gaps, err, calib = read(R / 'test_fairness_groups.csv'), pd.read_csv(R / 'test_fairness_gaps.csv', index_col=[0, 1]), read(R / 'test_error_analysis.csv'), read(R / 'test_calibration.csv')
expl = json.load(open(R / 'test_explain.json')); crit = read(R / 'stage1_criteria_results.csv'); exp = read(R / 'experiments.csv')
best = json.load(open(R / 'best_params.json')); blend = json.load(open(R / 'best_blend_alpha.json')); choice = json.load(open(R / 'mitigation_choice.json')); lat = json.load(open(R / 'latency.json'))
grid_n = {k: len(read(R / f'grid_{k}.csv')) for k in ['bias', 'item_cf', 'user_cf', 'svd']}
grid_best = {k: read(R / f'grid_{k}.csv').rmse.iloc[0] for k in grid_n}
DEP, SAFE, DIVR = 'Item CF + popularity (deployed)', 'Deployed + MMR/exposure (safe)', 'Deployed + MMR/exposure (diverse)'


def md_table(path, start):
    lines = Path(path).read_text().splitlines(); i = next(k for k, l in enumerate(lines) if l.startswith(start)); out = []
    for l in lines[i:]:
        if l.startswith('|'): out.append([c.strip() for c in l.strip('|').split('|')])
        elif out: break
    return [out[0]] + out[2:]


def sg(a, b, met):
    r = sig[(sig.comparison == f'{a} - {b}') & sig.metric.str.startswith(met)].iloc[0]; return r.estimate, r.ci_low, r.ci_high


def ci_txt(a, b, met, d=4):
    e, lo, hi = sg(a, b, met); return f'{e:+.{d}f} (95% CI {lo:+.{d}f} to {hi:+.{d}f})'


ild = lambda m: (div.loc[m, 'ild_genre'] + div.loc[m, 'ild_language']) / 2
n_tests = len([l for l in subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q', 'tests'], capture_output=True, text=True).stdout.splitlines() if '::' in l])
reg = ['Tamil', 'Telugu', 'Kannada', 'Malayalam']

facts = dict(
    raw_rows=len(raw), raw_users=raw.user_id.nunique(), raw_titles=raw.movie_title.nunique(), raw_dup=int(raw.duplicated(['user_id', 'movie_title']).sum()),
    rows=len(ds), cols=ds.shape[1], users=ds.user_id.nunique(), movies=ds.movie_id.nunique(), catalogue=len(cat), density=pc(len(ds) / (ds.user_id.nunique() * ds.movie_id.nunique())),
    pu_min=int(pu.min()), pu_med=int(pu.median()), pu_max=int(pu.max()), pu_mean=f'{pu.mean():.1f}', pi_min=int(pi.min()), pi_med=int(pi.median()), pi_max=int(pi.max()),
    r_mean=f'{ds.rating.mean():.2f}', r_med=f'{ds.rating.median():.1f}', r_std=f'{ds.rating.std():.2f}', r_ge4=pc((ds.rating >= 4).mean()),
    orig=int(ds.is_original.sum()), orig_pct=pc(ds.is_original.mean()), gen=int((~ds.is_original).sum()), t_min=ds.timestamp.min()[:10], t_max=ds.timestamp.max()[:10],
    n_train_core=len(tc), n_val=len(va), n_train_full=len(tf), n_test=len(te), n_test_users=int(rank.users_scored.iloc[0]), n_tests=n_tests,
    alpha=blend['item_cf'], lam_i=best['lam']['lam_i'], lam_u=best['lam']['lam_u'],
    rmse_item_cf=f4(rate.loc['Item CF', 'RMSE']), rmse_item_mean=f4(rate.loc['Item mean', 'RMSE']), rmse_global=f4(rate.loc['Global mean', 'RMSE']), rmse_bias=f4(rate.loc['Bias baseline', 'RMSE']),
    rmse_svd=f4(rate.loc['SVD', 'RMSE']), rmse_user_cf=f4(rate.loc['User CF', 'RMSE']),
    d_itemmean=ci_txt('Item CF', 'Item mean', 'RMSE'), d_global=ci_txt('Item CF', 'Global mean', 'RMSE'), d_bias=ci_txt('Item CF', 'Bias baseline', 'RMSE'),
    p10_dep=f4(rank.loc[DEP, 'P@10']), p10_pop=f4(rank.loc['Popularity', 'P@10']), p10_item=f4(rank.loc['Item CF', 'P@10']), p10_rand=f4(rank.loc['Random', 'P@10']), p10_safe=f4(rank.loc[SAFE, 'P@10']), p10_div=f4(rank.loc[DIVR, 'P@10']),
    d_dep_pop=ci_txt(DEP, 'Popularity', 'P@10'), d_item_pop=ci_txt('Item CF', 'Popularity', 'P@10'), d_dep_item=ci_txt(DEP, 'Item CF', 'P@10'), d_safe_dep=ci_txt(SAFE, DEP, 'P@10'),
    cov_dep=pc(rank.loc[DEP, 'Coverage@10']), cov_safe=pc(rank.loc[SAFE, 'Coverage@10']), cov_div=pc(rank.loc[DIVR, 'Coverage@10']), cov_pop=pc(rank.loc['Popularity', 'Coverage@10']),
    ild_dep=f3(ild(DEP)), ild_safe=f3(ild(SAFE)), ild_div=f3(ild(DIVR)), loss_safe=pc(1 - rank.loc[SAFE, 'P@10'] / rank.loc[DEP, 'P@10']), loss_div=pc(1 - rank.loc[DIVR, 'P@10'] / rank.loc[DEP, 'P@10']),
    hit_dep=pc(rank.loc[DEP, 'Hit@10']), hit_rand=pc(rank.loc['Random', 'Hit@10']), tail_dep=int(div.loc[DEP, 'tail_items_recommended']), tail_safe=int(div.loc[SAFE, 'tail_items_recommended']),
    tailshare_dep=pc(div.loc[DEP, 'tail_share_of_recs']), gini_dep=f'{div.loc[DEP, "gini_exposure"]:.2f}', gini_safe=f'{div.loc[SAFE, "gini_exposure"]:.2f}',
    align_dep=f3(div.loc[DEP, 'taste_alignment']), align_rand=f3(div.loc['Random', 'taste_alignment']),
    svd_r2=f3(expl['svd_item_factors_R2_on_genre_language']), svd_null=f3(expl['permutation_null_R2']),
    reg_cat=pc(lang.loc[reg, 'Catalogue'].sum()), reg_rel=pc(lang.loc[reg, 'Relevant test ratings'].sum()), reg_dep=pc(lang.loc[reg, DEP].sum()), reg_safe=pc(lang.loc[reg, SAFE].sum()),
    tel_dep=pc(lang.loc['Telugu', DEP]), tel_cat=pc(lang.loc['Telugu', 'Catalogue']), eng_dep=pc(lang.loc['English', DEP]), eng_cat=pc(lang.loc['English', 'Catalogue']),
    gap_act=f'{gaps.loc[("activity (train ratings)", "Item CF"), "RMSE gap (max-min)"]:.3f}', gap_taste=f'{gaps.loc[("generator taste group", "Item CF"), "RMSE gap (max-min)"]:.3f}', gap_taste_bias=f'{gaps.loc[("generator taste group", "Bias baseline"), "RMSE gap (max-min)"]:.3f}',
    lat_med=f'{lat["recommend k=10 (deployed)"]["median_ms"]:.1f}', lat_p95=f'{lat["recommend k=10 (deployed)"]["p95_ms"]:.1f}', lat_n=lat['recommend k=10 (deployed)']['n'],
    lat_safe_med=f'{lat["recommend k=10 diversify=safe"]["median_ms"]:.1f}',
    lam_safe=choice['safe']['lam'], gam_safe=choice['safe']['gamma'], lam_div=choice['diverse']['lam'], gam_div=choice['diverse']['gamma'],
    orig_rmse=f'{err[(err.by == "is_original") & (err.group == "True")].RMSE_item_cf.iloc[0]:.3f}', gen_rmse=f'{err[(err.by == "is_original") & (err.group == "False")].RMSE_item_cf.iloc[0]:.3f}',
    orig_test_share=pc(err[(err.by == 'is_original') & (err.group == 'True')].n.iloc[0] / len(te)),
    rmse_item_noorig=f4(tmn[tmn.task == 'rating'].set_index('model').loc['Item CF', 'RMSE']),
    py=f'{sys.version_info.major}.{sys.version_info.minor}', taste_users=f"{int(fair[fair.grouping.str.startswith('generator')].users.min())}-{int(fair[fair.grouping.str.startswith('generator')].users.max())}",
    lc=md_table(D / 'stage9_diagnostics.md', '| Share of train_core'),
)
ab = err[err.by == 'rating_bucket'].set_index('group'); facts['err_low'] = f'{ab.loc["1-2", "mean_error"]:+.2f}'; facts['err_high'] = f'{ab.loc["4.5-5", "mean_error"]:+.2f}'

T = {}
T['origin'] = dict(head=['Source', 'Rows', 'Share', 'Notes'], rows=[['Provided dataset.csv (kept)', str(facts['orig']), facts['orig_pct'], f'{facts["raw_dup"]} duplicate user-movie pairs dropped from the original {facts["raw_rows"]}; ratings and dates unchanged'],
    ['Synthetic (generated)', str(facts['gen']), pc(1 - ds.is_original.mean()), 'Created by src/generate_data.py (seed 42); flagged is_original = False']], widths=[2700, 900, 900, 4526])
T['data_stats'] = dict(head=['Property', 'Value'], rows=[['Rows / columns', f'{facts["rows"]} / {facts["cols"]}'], ['Users / movies rated / catalogue', f'{facts["users"]} / {facts["movies"]} / {facts["catalogue"]}'], ['Density', facts['density']],
    ['Ratings per user (min / median / max)', f'{facts["pu_min"]} / {facts["pu_med"]} / {facts["pu_max"]}'], ['Ratings per movie (min / median / max)', f'{facts["pi_min"]} / {facts["pi_med"]} / {facts["pi_max"]}'],
    ['Rating mean / median / std', f'{facts["r_mean"]} / {facts["r_med"]} / {facts["r_std"]}'], ['Share of ratings >= 4.0', facts['r_ge4']], ['Time range', f'{facts["t_min"]} to {facts["t_max"]}'],
    ['Languages (movies)', ', '.join(f'{k} {v}' for k, v in cat.language.value_counts().items())], ['Genres (movies)', ', '.join(f'{k} {v}' for k, v in cat.genre.value_counts().items())]], widths=[3800, 5226])
T['cleaning'] = dict(head=['Step', 'Rows removed', 'Reason'], rows=[['Standardise text, coerce types', '0', 'Trim spaces, canonical genre/language spelling'], ['Drop missing or unparseable key fields', '0', 'Ratings and identifiers must not be imputed'],
    ['Drop ratings outside 1-5 or not in 0.5 steps', '0', 'Valid rating scale'], ['Attach film attributes from catalogue', '0', 'One fixed genre, language and year per film'], ['Drop ratings dated before release year', '0', 'Anachronistic ratings are impossible'],
    ['Drop duplicate (user, movie) pairs', '0', 'One cell per user-movie pair'], ['Outlier check', '0', 'Bounded scale; valid extremes kept']], widths=[3700, 1300, 4026])
T['splits'] = dict(head=['Set', 'Ratings', 'Use'], rows=[['train_core', str(facts['n_train_core']), 'Fit models while tuning'], ['validation', str(facts['n_val']), 'Latest 20% of each user\'s train_full ratings; model selection'],
    ['train_full', str(facts['n_train_full']), 'Final fit of the selected models (train_core + validation)'], ['test', str(facts['n_test']), 'Latest 20% of each user\'s ratings; opened once in the final evaluation']], widths=[1800, 1300, 5926])
T['hyp'] = dict(head=['ID', 'Hypothesis', 'Test'], rows=[['H1', 'Movies with more ratings have higher mean rating', 'Spearman correlation'], ['H2', 'Ratings differ across languages', 'Kruskal-Wallis'], ['H3', 'Ratings differ across genres', 'Kruskal-Wallis'],
    ['H4', 'Users differ in how generously they rate', 'One-way ANOVA (omega-squared)'], ['H5', 'Users\' language mix is less varied than popularity alone predicts', 'Permutation test (300 simulations)']], widths=[700, 5600, 2726])
st = exp[exp.exp_id.between('E01', 'E15')]
T['val_exp'] = dict(head=['ID', 'Model', 'RMSE', 'MAE', 'P@10', 'R@10', 'Cov.@10'], rows=[[r.exp_id, r.model, f3(r.RMSE), f3(r.MAE), f3(r['P@10']), f3(r['R@10']), f3(r['Coverage@10'])] for _, r in st.iterrows()], widths=[700, 3500, 900, 900, 900, 900, 1226])
T['tuning'] = dict(head=['Model', 'Configs', 'Best settings', 'Mean-fold RMSE'], rows=[
    ['Bias baseline', str(grid_n['bias']), f'lam_i={best["lam"]["lam_i"]}, lam_u={best["lam"]["lam_u"]}', f4(grid_best['bias'])],
    ['Item-based CF', str(grid_n['item_cf']), ', '.join(f'{k}={v}' for k, v in best['item_cf'].items()), f4(grid_best['item_cf'])],
    ['User-based CF', str(grid_n['user_cf']), ', '.join(f'{k}={v}' for k, v in best['user_cf'].items()), f4(grid_best['user_cf'])],
    ['SVD (Surprise)', str(grid_n['svd']), ', '.join(f'{k}={v}' for k, v in best['svd'].items()), f4(grid_best['svd'])]], widths=[1900, 1400, 4500, 1226])
T['lc'] = dict(head=facts['lc'][0], rows=facts['lc'][1:], widths=[1500, 900, 1400, 1300, 1100, 1450, 1376])
T['test_rating'] = dict(head=['Model', 'RMSE', 'MAE', 'RMSE without original rows'], rows=[[m, f4(r.RMSE), f4(r.MAE), f4(tmn[tmn.task == 'rating'].set_index('model').loc[m, 'RMSE'])] for m, r in rate.iterrows()], widths=[3600, 1500, 1500, 2426])
T['test_rank'] = dict(head=['Model', 'P@5', 'P@10', 'R@10', 'NDCG@10', 'Hit@10', 'Cov.@10'], rows=[[m, f3(r['P@5']), f3(r['P@10']), f3(r['R@10']), f3(r['NDCG@10']), f3(r['Hit@10']), f3(r['Coverage@10'])] for m, r in rank.iterrows()],
    widths=[3300, 800, 800, 800, 1000, 900, 1426])
keep = [('Item CF', 'Item mean', 'RMSE'), ('Item CF', 'Global mean', 'RMSE'), ('Item CF', 'Bias baseline', 'RMSE'), ('SVD', 'Bias baseline', 'RMSE'), (DEP, 'Popularity', 'P@10'), ('Item CF', 'Popularity', 'P@10'), (DEP, 'Item CF', 'P@10'), (SAFE, DEP, 'P@10')]
nm = lambda s: s.replace(DEP, 'Deployed blend').replace(SAFE, 'Safe re-ranked')
T['sig'] = dict(head=['Comparison (first minus second)', 'Metric', 'Difference', '95% CI'], rows=[[nm(f'{a} - {b}'), met, f'{sg(a, b, met)[0]:+.4f}', f'[{sg(a, b, met)[1]:+.4f}, {sg(a, b, met)[2]:+.4f}]'] for a, b, met in keep], widths=[3700, 1100, 1700, 2526])
T['crit'] = dict(head=['#', 'Criterion', 'Target', 'Result on test set', 'Met?'], rows=crit.values.tolist(), widths=[400, 2200, 1300, 3926, 1200])
order = ['Random', 'Popularity', 'Item CF', 'SVD', DEP, SAFE, DIVR]
T['div'] = dict(head=['Model', 'Genre ILD', 'Lang. ILD', 'Coverage', 'Gini', 'Tail share', 'Tail movies shown', 'Taste align.'], rows=[[nm(m).replace('Deployed + MMR/exposure', 'Deployed + re-ranker'), f3(div.loc[m, 'ild_genre']), f3(div.loc[m, 'ild_language']), f3(div.loc[m, 'coverage']),
    f3(div.loc[m, 'gini_exposure']), pc(div.loc[m, 'tail_share_of_recs']), f'{int(div.loc[m, "tail_items_recommended"])} / 101', f3(div.loc[m, 'taste_alignment'])] for m in order], widths=[2500, 850, 850, 950, 750, 1000, 1226, 900])
T['lang'] = dict(head=['Language', 'Catalogue', 'Relevant test', 'Popularity', 'Deployed', 'Safe re-ranker'], rows=[[l, pc(lang.loc[l, 'Catalogue']), pc(lang.loc[l, 'Relevant test ratings']), pc(lang.loc[l, 'Popularity']), pc(lang.loc[l, DEP]), pc(lang.loc[l, SAFE])] for l in lang.index],
    widths=[1700, 1400, 1500, 1400, 1400, 1626])
T['fair'] = dict(head=['Grouping', 'Group', 'Users', 'RMSE Item CF', 'RMSE bias baseline'], rows=[[r.grouping.split(' (')[0], str(r.group), str(r.users), f3(r['RMSE Item CF']), f3(r['RMSE Bias baseline'])] for _, r in fair.iterrows()], widths=[2400, 2400, 900, 1600, 1726])
sel = err[err.by.isin(['rating_bucket', 'pop_tercile', 'activity', 'language'])]
T['err'] = dict(head=['Grouped by', 'Group', 'Ratings', 'RMSE Item CF', 'RMSE bias', 'Mean error'], rows=[[r.by.replace('rating_bucket', 'actual rating').replace('pop_tercile', 'movie popularity'), str(r.group), str(r.n), f3(r.RMSE_item_cf), f3(r.RMSE_bias), f'{r.mean_error:+.2f}'] for _, r in sel.iterrows()], widths=[1800, 1900, 1000, 1500, 1300, 1526])
T['calib'] = dict(head=['Predicted rating bin', 'Ratings', 'Mean predicted', 'Mean actual'], rows=[[str(r.pred_bin), str(r.n), f3(r.mean_predicted), f3(r.mean_actual)] for _, r in calib.iterrows()], widths=[2600, 1500, 2300, 2626])
T['latency'] = dict(head=['Request (300 real HTTP calls, localhost)', 'Median', '95th percentile', 'Max'], rows=[[k, f'{v["median_ms"]:.1f} ms', f'{v["p95_ms"]:.1f} ms', f'{v["max_ms"]:.1f} ms'] for k, v in lat.items()], widths=[4300, 1400, 1800, 1526])
T['api'] = dict(head=['Endpoint', 'Purpose'], rows=[['GET /api/health', 'Status, counts, methods'], ['GET /api/recommend?user_id=&k=&method=&diversify=', 'Top-k unseen movies; unknown users get a popularity fallback'], ['GET /api/similar?movie=&n=', 'Most similar movies (item-based CF)'],
    ['GET /api/user/<id>/history', 'A user\'s training ratings'], ['GET /api/explain?user_id=&movie=', 'Prediction broken into biases and similar-movie adjustment']], widths=[4400, 4626])
T['stack'] = dict(head=['Layer', 'Technology'], rows=[['Language / data', f'Python {sys.version_info.major}.{sys.version_info.minor}, pandas, NumPy, SciPy'], ['Models', 'Surprise (SVD); own NumPy implementations of item-based and user-based CF, baselines, popularity blend, MMR re-ranker'],
    ['Back end', 'Flask REST API; shared service layer (src/serving.py); joblib model files'], ['Front end', 'Streamlit dashboard'], ['Analysis and testing', 'Jupyter, matplotlib, seaborn, pytest (' + str(n_tests) + ' tests)'], ['Storage', 'CSV files (no database server was needed)']], widths=[2200, 6826])

insights = []
for blk in re.split(r'\n\n(?=\d+\. )', Path(D / 'eda_insights.md').read_text().split('\n\n', 1)[1].strip()):
    a = re.search(r'\*\*Insight:\*\* (.*?)\n\s+\*\*Decision:\*\* (.*)', blk, re.S); insights.append(dict(insight=a.group(1).strip(), decision=a.group(2).strip()))

figs = {}
for n in ['eda_01_rating_distribution', 'eda_02_long_tail', 'eda_03_matrix_sparsity', 'eda_04_genre_language_share', 'eda_06_ratings_over_time', 'eda_07_user_movie_bias', 's10_01_rmse_differences', 's10_02_p10_differences',
          's10_03_accuracy_diversity_tradeoff', 's10_04_language_share', 's10_05_exposure', 's10_06_error_analysis', 's10_07_fairness_groups', 'arch_dfd_level0', 'arch_block_diagram']:
    w, h = Image.open(F / f'{n}.png').size; figs[n] = dict(path=str((F / f'{n}.png').resolve()), w=w, h=h)

# real API sample for the report
sys.path.insert(0, '.')
from api.app import create_app
c = create_app().test_client(); samp = c.get('/api/recommend?user_id=8&k=3&diversify=safe').get_json()
sample = json.dumps({k: samp[k] for k in ['user_id', 'method', 'diversify', 'source']} | {'recommendations': [{k: r[k] for k in ['rank', 'title', 'genre', 'language', 'predicted_rating']} for r in samp['recommendations']], 'diversity': samp['diversity']}, indent=2)
err_sample = json.dumps(c.get('/api/recommend?user_id=abc').get_json(), indent=2)

Path('reports').mkdir(exist_ok=True)
json.dump(dict(facts=facts, tables=T, insights=insights, figs=figs, api_sample=sample, api_error=err_sample, hyp_results=[i['insight'] for i in insights[7:]]), open('reports/report_data.json', 'w'), indent=1, default=str)
print('ok', len(json.dumps(facts)), 'bytes of facts;', len(T), 'tables;', len(insights), 'insights;', n_tests, 'tests')
