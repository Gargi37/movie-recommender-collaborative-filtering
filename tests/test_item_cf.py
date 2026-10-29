import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, pytest
from src.baselines import BiasBaseline, GlobalMean
from src.item_cf import ItemCF
from src.data import load_clean, make_splits

@pytest.fixture(scope='module')
def parts():
    tc, v, tf, te = make_splits(load_clean())
    return tc, v

def test_similarity_properties(parts):
    m = ItemCF().fit(parts[0])
    assert np.allclose(m.sim, m.sim.T) and np.all(np.diag(m.sim) == 0) and np.abs(m.sim).max() <= 1

def test_predictions_in_range(parts):
    tc, v = parts; p = ItemCF(damp=1.0).fit(tc).predict(v)
    assert len(p) == len(v) and p.min() >= 1 and p.max() <= 5 and not np.isnan(p).any()

def test_recommend_excludes_seen_and_has_k(parts):
    tc, _ = parts; m = ItemCF(damp=1.0).fit(tc); u = tc.user_id.iloc[0]
    rec = m.recommend(u, k=10)
    assert len(rec) == 10 and not set(rec.movie_id) & set(tc[tc.user_id == u].movie_id)
    assert (rec.score.diff().dropna() <= 1e-9).all()          # sorted descending

def test_unknown_user_raises(parts):
    with pytest.raises(KeyError):
        ItemCF().fit(parts[0]).recommend(-1)

def test_huge_damping_equals_bias_baseline(parts):
    tc, v = parts
    assert np.allclose(ItemCF(damp=1e12).fit(tc).predict(v), BiasBaseline().fit(tc).predict(v))

def test_cold_start_policy(parts):
    tc, _ = parts; m = GlobalMean().fit(tc); ub, ib = tc.user_id.iloc[0], tc.movie_id.iloc[0]
    df = pd.DataFrame({'user_id': [ub, -1, -1], 'movie_id': ['NEW', ib, 'NEW']})
    p = ItemCF(damp=1.0).fit(tc).predict(df)
    icf = ItemCF(damp=1.0).fit(tc)
    assert np.isclose(p[0], icf.user_mean[icf.u2i[ub]])                      # unseen movie -> user mean
    assert np.isclose(p[1], icf.item_mean[icf.i2i[ib]])                      # unseen user -> movie mean
    assert np.isclose(p[2], icf.mu)                                          # both unseen -> global mean
