"""Stage 10 - evaluation, validation, diversity/fairness audit.

  python -m src.stage10 --split val     debug + choose re-ranker operating points (fit train_core, score validation)
  python -m src.stage10 --split test    FINAL run (fit train_full, score test). Refuses to run twice unless --force.

Decisions fixed BEFORE the test run (see docs/model_selection.md): deployed model = Item CF + popularity blend (alpha from folds);
re-ranker operating points are chosen on validation by the rule in choose_operating_points()."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from src.baselines import GlobalMean, UserMean, ItemMean, BiasBaseline, RandomRec, PopularityRec, TopRatedRec
from src.item_cf import ItemCF
from src.user_cf import UserCF
from src.svd import SVDMF
from src.hybrid import BlendRec
from src.audit import (ranked_lists, rank_metrics, diversity, share_by_code, boot_weights, ci, per_user_sse, boot_rmse_diff, diversify)
from src.generate_data import TASTE, GENRES, LANGS

ap = argparse.ArgumentParser(); ap.add_argument('--split', choices=['val', 'test'], required=True); ap.add_argument('--force', action='store_true')
args = ap.parse_args(); SPLIT = args.split
TEST_SHA = '57d5dcad96de4b2babc3f855220b1a81daa7ec63d927acad78ce8f1f6d59f7bc'
Path('results').mkdir(exist_ok=True); Path('reports/figures').mkdir(parents=True, exist_ok=True)
read = lambda n: pd.read_csv(f'data/processed/split/{n}.csv', parse_dates=['timestamp'])
cat = pd.read_csv('data/raw/movie_catalogue.csv'); taste_groups = pd.read_csv('data/raw/synthetic_taste_groups.csv')
P = json.load(open('results/best_params.json')); lam = P['lam']; ALPHA = json.load(open('results/best_blend_alpha.json'))['item_cf']
if SPLIT == 'test':
    assert hashlib.sha256(Path('data/processed/split/test.csv').read_bytes()).hexdigest() == TEST_SHA, 'test set changed since it was locked!'
    if Path('results/test_opened.txt').exists() and not args.force:
        sys.exit('The test set was already opened once (results/test_opened.txt). Use --force only if you must, and disclose it.')
    train, target = read('train_full'), read('test')
else:
    train, target = read('train_core'), read('val')
SUF = SPLIT

# ------------------------------------------------------------------ models
specs = {
    'Global mean': GlobalMean, 'User mean': UserMean, 'Item mean': ItemMean, 'Bias baseline': lambda: BiasBaseline(**lam),
    'Random': lambda: RandomRec(0), 'Popularity': PopularityRec, 'Top-rated': TopRatedRec,
    'Item CF': lambda: ItemCF(**P['item_cf'], **lam), 'User CF': lambda: UserCF(**P['user_cf'], **lam), 'SVD': lambda: SVDMF(**P['svd']),
    'Item CF + popularity (deployed)': lambda: BlendRec(ItemCF(**P['item_cf'], **lam), alpha=ALPHA),
    'SVD + popularity': lambda: BlendRec(SVDMF(**P['svd']), alpha=json.load(open('results/best_blend_alpha.json'))['svd']),
}
M = {k: f().fit(train) for k, f in specs.items()}
ref = M['Item CF + popularity (deployed)']
if SPLIT == 'test':
    import joblib
    saved = joblib.load('models/item_cf_blend.joblib')
    assert np.allclose(saved._S, ref._S), 'refit deployed model differs from the saved one'
nU, nI = ref.nU, ref.nI
item_meta = cat.set_index('movie_id').loc[ref.item_ids]
gcode = np.array([GENRES.index(g) for g in item_meta.genre]); lcode = np.array([LANGS.index(l) for l in item_meta.language])
RATE = ['Global mean', 'User mean', 'Item mean', 'Bias baseline', 'Item CF', 'User CF', 'SVD']
RANK = ['Popularity', 'Top-rated', 'Item CF', 'User CF', 'SVD', 'Item CF + popularity (deployed)', 'SVD + popularity']
Sref, seen = ref._S, ref.seen

# user-side helpers for audit
cnt_g = np.zeros((nU, len(GENRES))); np.add.at(cnt_g, (ref.ui, gcode[ref.ii]), 1)
top_genres = {u: set(np.argsort(-cnt_g[u], kind='stable')[:2]) for u in range(nU)}
relM = np.zeros((nU, nI), bool)
tr_ = target[target.rating >= 4.0]; ok = tr_.user_id.isin(ref.u2i) & tr_.movie_id.isin(ref.i2i)
relM[tr_[ok].user_id.map(ref.u2i).astype(int), tr_[ok].movie_id.map(ref.i2i).astype(int)] = True
names = list(TASTE)
aff_table = np.array([[TASTE[n][0].get(GENRES[gcode[i]], 0) + TASTE[n][1].get(LANGS[lcode[i]], 0) for i in range(nI)] for n in names])
ug = taste_groups.set_index('user_id').primary_taste_group.reindex(ref.user_ids).map({n: i for i, n in enumerate(names)}).to_numpy()


def audit_lists(lists):
    d = diversity(lists, ref, gcode, lcode, 10, user_top_genres=top_genres, hits=np.take_along_axis(relM, lists[:, :10], 1))
    d['taste_alignment'] = float(aff_table[ug[:, None], lists[:, :10]].mean())
    return d


# ------------------------------------------------------------------ re-ranker operating points (VALIDATION only)
def choose_operating_points():
    grid = []
    for lam_, gam in [(l, g) for l in [1.0, 0.8, 0.6, 0.4] for g in [0, 0.5, 1, 2, 4]]:
        lists = diversify(Sref, seen, gcode, lcode, lam_, gam, 50, 10)
        rm = rank_metrics(lists, ref, target, 10); d = audit_lists(lists)
        grid.append(dict(lam=lam_, gamma=gam, P10=rm.p.mean(), NDCG10=rm.ndcg.mean(), **d))
    g = pd.DataFrame(grid); base = g[(g.lam == 1.0) & (g.gamma == 0)].iloc[0]
    g['P10_rel_loss'] = 1 - g.P10 / base.P10
    g['ild_combined'] = (g.ild_genre + g.ild_language) / 2
    def pick(max_loss):
        c = g[g.P10_rel_loss <= max_loss + 1e-12].sort_values(['coverage', 'ild_combined'], ascending=False)
        r = c.iloc[0]; return dict(lam=float(r.lam), gamma=float(r.gamma))
    choice = dict(safe=pick(0.05), diverse=pick(0.20), rule='maximise coverage@10 (tie: combined ILD) subject to relative P@10 loss <= 5% (safe) / <= 20% (diverse), chosen on validation')
    g.to_csv('results/mitigation_grid_val.csv', index=False)
    json.dump(choice, open('results/mitigation_choice.json', 'w'), indent=2)
    return g, choice


if SPLIT == 'val':
    g, choice = choose_operating_points()
    print(g.round(3).to_string(index=False)); print(json.dumps(choice, indent=2))
choice = json.load(open('results/mitigation_choice.json'))
L = {n: ranked_lists(M[n]._S, M[n].seen, 50) for n in RANK}
L['Random'] = ranked_lists(M['Random']._S, M['Random'].seen, 50)
L['Deployed + MMR/exposure (safe)'] = diversify(Sref, seen, gcode, lcode, choice['safe']['lam'], choice['safe']['gamma'], 50, 10)
L['Deployed + MMR/exposure (diverse)'] = diversify(Sref, seen, gcode, lcode, choice['diverse']['lam'], choice['diverse']['gamma'], 50, 10)
RANK_ALL = RANK + ['Deployed + MMR/exposure (safe)', 'Deployed + MMR/exposure (diverse)']

# ------------------------------------------------------------------ main metrics
def main_table(T):
    rows = []
    for n in RATE:
        p = M[n].predict(T); e = T.rating.to_numpy() - p
        rows.append(dict(model=n, task='rating', RMSE=np.sqrt((e ** 2).mean()), MAE=np.abs(e).mean()))
    out = pd.DataFrame(rows)
    rk = []
    for n in ['Random'] + RANK_ALL:
        lists_list = [ranked_lists(RandomRec(s).fit(train)._S, ref.seen, 50) for s in range(50)] if n == 'Random' else [L[n]]
        vals = []
        for lists in lists_list:
            r5, r10 = rank_metrics(lists, ref, T, 5), rank_metrics(lists, ref, T, 10)
            vals.append([r5.p.mean(), r10.p.mean(), r5.r.mean(), r10.r.mean(), r5.ndcg.mean(), r10.ndcg.mean(), r10.hit.mean(), diversity(lists, ref, gcode, lcode, 10)['coverage']])
        v = np.mean(vals, 0)
        rk.append(dict(model=n, task='ranking', **dict(zip(['P@5', 'P@10', 'R@5', 'R@10', 'NDCG@5', 'NDCG@10', 'Hit@10', 'Coverage@10'], v)), users_scored=len(r10)))
    return pd.concat([out, pd.DataFrame(rk)], ignore_index=True)

res_all = main_table(target)
res_noorig = main_table(target[~target.is_original])
res_all.to_csv(f'results/{SUF}_metrics.csv', index=False); res_noorig.to_csv(f'results/{SUF}_metrics_no_original.csv', index=False)

# ------------------------------------------------------------------ paired bootstrap over users
Wr = boot_weights(nU, 2000, 42)
sse = {n: per_user_sse(target.user_id, target.rating, M[n].predict(target), ref.user_ids) for n in RATE}
cnt_u = sse['Item CF'][1]
sig = []
for a, b in [('Item CF', 'Item mean'), ('Item CF', 'Global mean'), ('Item CF', 'Bias baseline'), ('User CF', 'Bias baseline'), ('SVD', 'Bias baseline'), ('SVD', 'Item CF')]:
    d, ra, rb = boot_rmse_diff(sse[a][0], sse[b][0], cnt_u, Wr); lo, hi = ci(d)
    pa = np.sqrt(sse[a][0].sum() / cnt_u.sum()) - np.sqrt(sse[b][0].sum() / cnt_u.sum())
    sig.append(dict(comparison=f'{a} - {b}', metric='RMSE (negative = first is better)', estimate=pa, ci_low=lo, ci_high=hi, share_boot_below_0=float((d < 0).mean())))
RM = {n: rank_metrics(L[n], ref, target, 10) for n in RANK_ALL}
Wk = boot_weights(len(RM['Popularity']), 2000, 43)
for a, b in [('Item CF + popularity (deployed)', 'Popularity'), ('Item CF', 'Popularity'), ('Item CF + popularity (deployed)', 'Item CF'), ('SVD + popularity', 'Popularity'),
             ('Deployed + MMR/exposure (safe)', 'Item CF + popularity (deployed)'), ('Deployed + MMR/exposure (diverse)', 'Item CF + popularity (deployed)')]:
    for met in ['p', 'ndcg']:
        diff = RM[a][met].to_numpy() - RM[b][met].to_numpy(); d = Wk @ diff / len(diff); lo, hi = ci(d)
        sig.append(dict(comparison=f'{a} - {b}', metric={'p': 'P@10', 'ndcg': 'NDCG@10'}[met] + ' (positive = first is better)', estimate=float(diff.mean()), ci_low=lo, ci_high=hi, share_boot_below_0=float((d < 0).mean())))
sig = pd.DataFrame(sig); sig.to_csv(f'results/{SUF}_significance.csv', index=False)

# ------------------------------------------------------------------ diversity / exposure audit
div = pd.DataFrame({n: audit_lists(L[n]) for n in ['Random'] + RANK_ALL}).T
div.to_csv(f'results/{SUF}_diversity_audit.csv')
lang_share = pd.DataFrame({'Catalogue': np.bincount(lcode, minlength=len(LANGS)) / nI,
                           'Train ratings': np.bincount(lcode[ref.ii], minlength=len(LANGS)) / len(ref.ii),
                           'Relevant test ratings': np.bincount(lcode[np.nonzero(relM)[1]], minlength=len(LANGS)) / relM.sum(),
                           **{n: share_by_code(L[n], lcode, len(LANGS)) for n in ['Popularity', 'Item CF', 'SVD', 'Item CF + popularity (deployed)', 'Deployed + MMR/exposure (safe)', 'Deployed + MMR/exposure (diverse)']}}, index=LANGS)
lang_share.to_csv(f'results/{SUF}_language_share.csv')

# ------------------------------------------------------------------ fairness across user groups
act = pd.qcut(pd.Series(ref.n_user).rank(method='first'), 3, labels=['light', 'medium', 'heavy']).to_numpy()
gen = pd.qcut(pd.Series(ref.user_mean).rank(method='first'), 3, labels=['strict raters', 'middle', 'generous raters']).to_numpy()
tg = np.array([names[i] if i == i else 'unknown' for i in np.nan_to_num(ug, nan=-1).astype(int)]) if False else np.array([names[int(i)] for i in ug])
user_groups = {'activity (train ratings)': act, 'rating generosity': gen, 'generator taste group': tg}
fair = []; rk_users = RM['Popularity'].index.to_numpy(); u_idx_rank = np.array([ref.u2i[u] for u in rk_users])
for gname, lab in user_groups.items():
    for lv in pd.unique(lab):
        m_ = lab == lv
        row = dict(grouping=gname, group=lv, users=int(m_.sum()), test_ratings=int(cnt_u[m_].sum()))
        for n in ['Item CF', 'Bias baseline', 'SVD']:
            row[f'RMSE {n}'] = float(np.sqrt(sse[n][0][m_].sum() / cnt_u[m_].sum()))
        mk = m_[u_idx_rank]
        for n in ['Item CF + popularity (deployed)', 'Deployed + MMR/exposure (safe)']:
            row[f'P@10 {n}'] = float(RM[n].p.to_numpy()[mk].mean()) if mk.any() else float('nan')
            row[f'NDCG@10 {n}'] = float(RM[n].ndcg.to_numpy()[mk].mean()) if mk.any() else float('nan')
        fair.append(row)
fair = pd.DataFrame(fair); fair.to_csv(f'results/{SUF}_fairness_groups.csv', index=False)
gaps = {}
for gname, lab in user_groups.items():
    lv = list(pd.unique(lab)); masks = [lab == v for v in lv]
    for n in ['Item CF', 'Bias baseline']:
        gr = np.stack([np.sqrt(Wr @ (sse[n][0] * m_) / (Wr @ (cnt_u * m_) + 1e-12)) for m_ in masks])
        bg = gr.max(0) - gr.min(0)
        point = max(np.sqrt(sse[n][0][m_].sum() / cnt_u[m_].sum()) for m_ in masks) - min(np.sqrt(sse[n][0][m_].sum() / cnt_u[m_].sum()) for m_ in masks)
        gaps[(gname, n)] = (float(point), *ci(bg))
gaps = pd.DataFrame(gaps, index=['RMSE gap (max-min)', 'ci_low', 'ci_high']).T; gaps.to_csv(f'results/{SUF}_fairness_gaps.csv')

# ------------------------------------------------------------------ error analysis (deployed rating model = Item CF)
T = target.copy(); T['pred'] = M['Item CF'].predict(T); T['pred_bias'] = M['Bias baseline'].predict(T); T['pred_gm'] = M['Global mean'].predict(T)
T['pop_tercile'] = pd.qcut(T.movie_id.map(dict(zip(ref.item_ids, ref.n_item))).rank(method='first'), 3, labels=['rare movies', 'mid', 'popular movies'])
T['activity'] = T.user_id.map(dict(zip(ref.user_ids, act)))
T['rating_bucket'] = pd.cut(T.rating, [0, 2, 3, 4, 5], labels=['1-2', '2.5-3', '3.5-4', '4.5-5'])
T['era'] = pd.cut(T.release_year, [1900, 2009, 2017, 2100], labels=['before 2010', '2010-2017', '2018 or later'])
err = []
for col in ['language', 'genre', 'pop_tercile', 'activity', 'rating_bucket', 'era', 'is_original']:
    for lv, g in T.groupby(col, observed=True):
        rm_ = lambda c: float(np.sqrt(((g.rating - g[c]) ** 2).mean()))
        err.append(dict(by=col, group=str(lv), n=len(g), RMSE_item_cf=rm_('pred'), RMSE_bias=rm_('pred_bias'), RMSE_global_mean=rm_('pred_gm'), mean_error=float((g.pred - g.rating).mean())))
err = pd.DataFrame(err); err.to_csv(f'results/{SUF}_error_analysis.csv', index=False)
T['pred_bin'] = pd.cut(T.pred, [0, 3.0, 3.5, 4.0, 4.5, 5.01], right=False, labels=['<3.0', '3.0-3.5', '3.5-4.0', '4.0-4.5', '4.5+'])
calib = T.groupby('pred_bin', observed=True).agg(n=('rating', 'size'), mean_predicted=('pred', 'mean'), mean_actual=('rating', 'mean')).reset_index()
calib.to_csv(f'results/{SUF}_calibration.csv', index=False)

# ------------------------------------------------------------------ explanation (what drives the model)
bi = pd.Series(ref.base.bi, index=ref.item_ids).sort_values(); title = cat.set_index('movie_id').movie_title
bias_tbl = pd.concat([bi.head(5).rename('movie_bias').to_frame().assign(movie=lambda d: d.index.map(title)), bi.tail(5).rename('movie_bias').to_frame().assign(movie=lambda d: d.index.map(title))])
svd = M['SVD']; Q = svd.Q; X = pd.get_dummies(pd.DataFrame({'g': item_meta.genre.to_numpy(), 'l': item_meta.language.to_numpy()}), drop_first=True).to_numpy(float)
X1 = np.c_[np.ones(len(X)), X]
def r2(Y, X1):
    beta = np.linalg.lstsq(X1, Y, rcond=None)[0]; return 1 - ((Y - X1 @ beta) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
obs_r2 = r2(Q, X1); rng = np.random.default_rng(42)
null_r2 = np.mean([r2(Q, X1[rng.permutation(len(X1))]) for _ in range(200)])
expl = dict(svd_item_factors_R2_on_genre_language=float(obs_r2), permutation_null_R2=float(null_r2))
json.dump(expl, open(f'results/{SUF}_explain.json', 'w'))

print(f'\n=== {SPLIT.upper()} split: train {len(train)} rows, target {len(target)} rows ===')
pd.set_option('display.width', 250, 'display.max_columns', 40)
print(res_all.round(4).to_string(index=False)); print('\n-- excluding is_original --'); print(res_noorig.round(4).to_string(index=False))
print('\n-- significance --'); print(sig.round(4).to_string(index=False))
print('\n-- diversity --'); print(div.round(3).to_string())
print('\n-- language share --'); print((lang_share * 100).round(1).to_string())
print('\n-- fairness gaps --'); print(gaps.round(4).to_string()); print('\n-- error analysis --'); print(err.round(3).to_string(index=False)); print('\n-- calibration --'); print(calib.round(3).to_string(index=False))
print('\nSVD factor R2 on metadata', expl)
pd.to_pickle(dict(res_all=res_all, res_noorig=res_noorig, sig=sig, div=div, lang_share=lang_share, fair=fair, gaps=gaps, err=err, calib=calib, bias_tbl=bias_tbl, expl=expl, choice=choice,
                  n_train=len(train), n_target=len(target), n_users=nU, n_items=nI), f'results/{SUF}_bundle.pkl')
if SPLIT == 'test':
    Path('results/test_opened.txt').write_text(f'test set opened for the final evaluation at {time.strftime("%Y-%m-%d %H:%M:%S")}; sha256 {TEST_SHA}\n')
