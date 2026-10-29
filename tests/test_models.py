import sys; sys.path.insert(0, '.')
import re, pathlib
import numpy as np, pandas as pd, pytest
from src.data import load_clean, make_splits
from src.user_cf import UserCF
from src.svd import SVDMF
from src.item_cf import ItemCF
from src.hybrid import BlendRec
from src.recommender import Recommender

@pytest.fixture(scope='module')
def parts():
    tc, v, tf, te = make_splits(load_clean()); return tc, v

def test_pearson_matches_numpy(parts):
    tc, _ = parts; m = UserCF(sim='pearson', shrink=0).fit(tc)
    R = np.zeros_like(m.B); R[m.ui, m.ii] = m.r; checked = 0
    for u in range(40):
        for v in range(u + 1, 80):
            both = (m.B[u] > 0) & (m.B[v] > 0)
            if both.sum() >= 4 and R[u, both].std() > 0 and R[v, both].std() > 0:
                assert np.isclose(m.sim[u, v], np.corrcoef(R[u, both], R[v, both])[0, 1], atol=1e-9); checked += 1
    assert checked > 0

def test_blend_alpha_zero_keeps_base_ranking(parts):
    tc, _ = parts; base = ItemCF(shrink=10, damp=1.0).fit(tc); u = tc.user_id.iloc[0]
    blend = BlendRec(ItemCF(shrink=10, damp=1.0), alpha=0).fit(tc)
    assert list(blend.recommend(u, 10).movie_id) == list(base.recommend(u, 10).movie_id)

def test_svd_range_and_determinism(parts):
    tc, v = parts; a = SVDMF(seed=1).fit(tc).predict(v); b = SVDMF(seed=1).fit(tc).predict(v)
    assert np.allclose(a, b) and a.min() >= 1 and a.max() <= 5

def test_deployed_recommender():
    rec = Recommender.load('models/final_recommender.joblib'); tf = pd.read_csv('data/processed/split/train_full.csv')
    u = int(tf.user_id.iloc[0]); out = rec.recommend(u, 10)
    assert len(out) == 10 and (out.source == 'personalised').all() and not set(out.movie_id) & set(tf[tf.user_id == u].movie_id)
    cold = rec.recommend(-999, 5)
    assert len(cold) == 5 and (cold.source == 'popularity_fallback').all()
    for bad in (0, -1, 'x', 2.5):
        with pytest.raises(ValueError):
            rec.recommend(u, bad)
    assert 1 <= rec.predict_rating(u, out.movie_id.iloc[0]) <= 5 and 1 <= rec.predict_rating(-999, 'NEW') <= 5
    assert len(rec.similar_movies(out.movie_id.iloc[0], 3)) == 3

def test_stage8_9_scripts_never_read_test_split():
    for f in ['run_stage8.py', 'tune.py', 'diagnostics.py', 'finalize_stage9.py', 'demo_item_cf.py']:
        code = pathlib.Path('src', f).read_text()
        assert not re.search(r"read\(\s*['\"]test['\"]\s*\)|split/test", code), f
