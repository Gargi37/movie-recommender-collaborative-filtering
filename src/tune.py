"""Stage 9 tuning. Two rolling, time-respecting folds built from TRAIN data only (test.csv is never read):
   fold 1: fit on the first 75% of each user's train_core ratings -> score the next 25%
   fold 2: fit on train_core -> score validation
Selection metric = mean RMSE over the two folds. Writes results/grid_*.csv and results/best_params.json."""
import itertools, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import per_user_split
from src.baselines import BiasBaseline
from src.item_cf import ItemCF
from src.user_cf import UserCF
from src.svd import SVDMF
from src.metrics import rating_eval, ranking_eval

read = lambda n: pd.read_csv(f'data/processed/split/{n}.csv', parse_dates=['timestamp'])
train_core, val = read('train_core'), read('val')
a, b = per_user_split(train_core, 0.25)
FOLDS = [(a, b), (train_core, val)]
Path('results').mkdir(exist_ok=True)


def evaluate(make):
    out = []
    for tr, te in FOLDS:
        m = make().fit(tr)
        r, k = rating_eval(m, te), ranking_eval(m, tr, te, 10)
        out.append((r['rmse'], r['mae'], k['precision'], k['recall']))
    o = np.array(out)
    return dict(rmse_f1=o[0, 0], rmse_f2=o[1, 0], rmse=o[:, 0].mean(), mae=o[:, 1].mean(), p10=o[:, 2].mean(), r10=o[:, 3].mean())


def grid(name, make, space):
    keys = list(space); rows = []; t = time.time()
    for vals in itertools.product(*space.values()):
        cfg = dict(zip(keys, vals)); rows.append({**cfg, **evaluate(lambda: make(**cfg))})
    df = pd.DataFrame(rows).sort_values('rmse').reset_index(drop=True)
    df.to_csv(f'results/grid_{name}.csv', index=False)
    print(f'{name}: {len(df)} configs in {time.time() - t:.0f}s'); print(df.head(5).round(4).to_string(index=False))
    best_cfg = {k: df.loc[0, k].item() if hasattr(df.loc[0, k], 'item') else df.loc[0, k] for k in keys}
    for k in keys:
        vals = space[k]
        if len(vals) > 2 and isinstance(vals[0], (int, float)) and best_cfg[k] in (min(vals), max(vals)):
            print(f'  !! {name}: best {k}={best_cfg[k]} is at the edge of the grid {vals}')
    n_tied = int((df.rmse - df.rmse.iloc[0] < 1e-6).sum()); print(f'  configs tied with the best (RMSE within 1e-6): {n_tied}\n')
    return df, best_cfg


from src.baselines import PopularityRec, TopRatedRec
def fold_rank(make):
    o = [ranking_eval(make().fit(tr), tr, te, 10) for tr, te in FOLDS]
    return round(np.mean([x['precision'] for x in o]), 4), round(np.mean([x['recall'] for x in o]), 4)
print('same-fold references  P@10 / R@10:  popularity', fold_rank(PopularityRec), '| top-rated', fold_rank(TopRatedRec), '\n')
best = {}
print('fold sizes: fold1', len(a), '->', len(b), '| fold2', len(train_core), '->', len(val), '\n')
g, best['bias'] = grid('bias', lambda **c: BiasBaseline(**c), dict(lam_i=[0, 1, 2, 5, 10, 20], lam_u=[0, 2, 5, 10, 20, 30]))
lam = dict(lam_i=best['bias']['lam_i'], lam_u=best['bias']['lam_u'])
base_row = evaluate(lambda: BiasBaseline(**lam)); print('tuned bias baseline:', {k: round(v, 4) for k, v in base_row.items()}, '\n')
g, best['item_cf'] = grid('item_cf', lambda **c: ItemCF(**c, **lam), dict(k=[10, 20, 40, 80, 126], shrink=[10, 25, 50, 100, 200, 500], damp=[0.1, 0.25, 0.5, 1, 2, 5]))
g, best['user_cf'] = grid('user_cf', lambda **c: UserCF(**c, **lam), dict(sim=['pearson', 'cosine'], k=[40, 80, 160, 309], shrink=[10, 25, 50, 100], damp=[2, 5, 10, 20, 50]))
g, best['svd'] = grid('svd', lambda **c: SVDMF(**c), dict(n_factors=[5, 10, 20, 50], lr=[0.002, 0.005, 0.01, 0.02], reg=[0.02, 0.05, 0.1, 0.2, 0.4], epochs=[10, 20, 50, 100]))
best['lam'] = lam
json.dump(best, open('results/best_params.json', 'w'), indent=2)
print(json.dumps(best))
