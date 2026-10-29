import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, pytest
from src import audit
from src.recommender import Recommender

class Dummy:                                   # minimal model stand-in
    u2i = {10: 0, 20: 1}; i2i = {'a': 0, 'b': 1, 'c': 2, 'd': 3}; nI = 4
    n_item = np.array([10., 5., 3., 2.])

def test_rank_metrics_hand_computed():
    lists = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    target = pd.DataFrame({'user_id': [10, 10, 20], 'movie_id': ['a', 'c', 'a'], 'rating': [5.0, 4.0, 4.5]})
    m = audit.rank_metrics(lists, Dummy, target, K=2)
    assert np.isclose(m.loc[10, 'p'], 0.5) and np.isclose(m.loc[10, 'r'], 0.5)       # hit a in top2, 2 relevant
    idcg = 1 + 1 / np.log2(3)
    assert np.isclose(m.loc[10, 'ndcg'], 1 / idcg)
    assert m.loc[20, 'p'] == 0 and m.loc[20, 'ndcg'] == 0

def test_gini_and_ild():
    assert np.isclose(audit.gini([1, 1, 1, 1]), 0) and audit.gini([0, 0, 0, 10]) > 0.7
    assert audit.ild(np.array([0, 0, 1, 1]), np.array([[0, 1]])) == 0.0
    assert audit.ild(np.array([0, 1, 2, 3]), np.array([[0, 1, 2, 3]])) == 1.0

def test_bootstrap_weights_sum_to_n():
    W = audit.boot_weights(50, 100); assert (W.sum(1) == 50).all()

def test_diversify_identity_and_effects():
    rng = np.random.default_rng(0); nU, nI = 30, 40
    S = rng.random((nU, nI)); seen = np.zeros((nU, nI), bool); seen[:, :3] = True
    g, l = rng.integers(0, 4, nI), rng.integers(0, 3, nI)
    plain = audit.ranked_lists(S, seen, 10)
    assert (audit.diversify(S, seen, g, l, lam=1.0, gamma=0.0, N=20, K=10) == plain).all()
    div = audit.diversify(S, seen, g, l, lam=0.5, gamma=0.0, N=20, K=10)
    assert audit.ild(g, div) >= audit.ild(g, plain) and not (seen[np.arange(nU)[:, None], div]).any()
    expo = audit.diversify(S, seen, g, l, lam=1.0, gamma=4.0, N=40, K=10)
    assert len(np.unique(expo)) >= len(np.unique(plain))

def test_explain_adds_up():
    rec = Recommender.load(); u = int(next(iter(rec.base.u2i))); m = rec.recommend(u, 1).movie_id.iloc[0]
    parts, nb = rec.explain(u, m)
    assert np.isclose(parts['predicted'], rec.predict_rating(u, m), atol=1e-6) and len(nb) <= 3
    with pytest.raises(KeyError):
        rec.explain(-1, m)
