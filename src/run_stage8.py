"""Stage 8: baselines + first item-based CF, fitted on train_core and scored on VALIDATION. The test set is never read."""
import numpy as np
import pandas as pd
from src.baselines import GlobalMean, UserMean, ItemMean, BiasBaseline, RandomRec, PopularityRec, TopRatedRec
from src.item_cf import ItemCF
from src.metrics import rating_eval, ranking_eval
from src.experiments import log

def read(name):
    return pd.read_csv(f'data/processed/split/{name}.csv', parse_dates=['timestamp'])
train, val = read('train_core'), read('val')          # NOTE: test.csv is intentionally not loaded
K = 10
rows = []
def add(exp_id, name, settings, rate=None, rank=None, notes=''):
    rows.append([exp_id, 8, name, settings, 'train_core', 'val', rate and rate['rmse'], rate and rate['mae'],
                 rank and rank['precision'], rank and rank['recall'], rank and rank['hit_rate'], rank and rank['coverage'], rank and rank['users'], notes])

for eid, name, m in [('E01', 'Global mean', GlobalMean()), ('E02', 'User mean', UserMean()), ('E03', 'Item mean', ItemMean()),
                     ('E04', 'Bias baseline (mu+b_u+b_i)', BiasBaseline())]:
    m.fit(train)
    add(eid, name, 'lam_i=5, lam_u=10' if eid == 'E04' else '-', rate=rating_eval(m, val),
        notes={'E01': 'Minimum standard for RMSE', 'E03': 'Stage 1 criterion: CF must beat this', 'E04': 'Strongest non-personalised rating baseline'}.get(eid, ''))

# ranking baselines
rs = [ranking_eval(RandomRec(seed=s).fit(train), train, val, K) for s in range(50)]
rand = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
add('E05', 'Random recommender', 'mean of 50 seeds', rank=rand, notes='Floor for ranking metrics')
add('E06', 'Popularity (most rated)', '-', rank=ranking_eval(PopularityRec().fit(train), train, val, K), notes='Stage 1 criterion: CF must beat this')
add('E07', 'Top-rated (shrunk mean)', 'm=10', rank=ranking_eval(TopRatedRec().fit(train), train, val, K))

cat = pd.read_csv('data/raw/movie_catalogue.csv')
icf = ItemCF(k=20, shrink=10, meta=cat).fit(train)
add('E08', 'Item-based CF (adjusted cosine, residuals)', 'k=20, shrink=10 (untuned)', rate=rating_eval(icf, val), rank=ranking_eval(icf, train, val, K),
    notes='First working module; tuned in Stage 9')
icf2 = ItemCF(k=20, shrink=10, damp=1.0, meta=cat).fit(train)
add('E09', 'Item-based CF + damping', 'k=20, shrink=10, damp=1.0 (untuned)', rate=rating_eval(icf2, val), rank=ranking_eval(icf2, train, val, K),
    notes='Fix for E08: damp in denominator; value is an assumption, tuned in Stage 9')
df = log(rows)
pd.set_option('display.width', 250)
print(df[['exp_id', 'model', 'RMSE', 'MAE', 'P@10', 'R@10', 'Hit@10', 'Coverage@10', 'users_scored']].round(3).to_string(index=False))
