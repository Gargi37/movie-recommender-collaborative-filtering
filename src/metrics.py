"""Evaluation metrics shared by all stages."""
import numpy as np
import pandas as pd

from src.data import RELEVANT


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def ranking_eval(model, train, target, K=10, relevant=RELEVANT):
    """Top-K ranking metrics.
    Candidates for a user = every movie the model knows minus the movies in that user's `train` rows.
    Relevant = the user's `target` ratings >= `relevant`. Unrated movies count as non-relevant (standard, but pessimistic).
    Only users with at least one relevant target movie are scored. Relevant movies unknown to the model count as misses.
    Returns precision@K, recall@K, hit rate@K, catalogue coverage@K and the number of users scored."""
    S = model.score_matrix()
    u2i, i2i = model.u2i, model.i2i
    seen = np.zeros(S.shape, bool)
    tu, ti = train.user_id.map(u2i), train.movie_id.map(i2i)
    ok = tu.notna() & ti.notna()
    seen[tu[ok].astype(int), ti[ok].astype(int)] = True
    rel = target[target.rating >= relevant]
    P, R, H, rec_items = [], [], [], set()
    order_tiebreak = np.arange(S.shape[1])
    for u, g in rel.groupby('user_id'):
        if u not in u2i:
            P.append(0.0); R.append(0.0); H.append(0.0); continue
        row = np.where(seen[u2i[u]], -np.inf, S[u2i[u]])
        top = np.lexsort((order_tiebreak, -row))[:K]            # score desc, ties by item index (deterministic)
        rec_items.update(top.tolist())
        rel_idx = {i2i[m] for m in g.movie_id if m in i2i}
        hits = len(rel_idx & set(top.tolist()))
        P.append(hits / K); R.append(hits / len(g)); H.append(float(hits > 0))
    return dict(precision=float(np.mean(P)), recall=float(np.mean(R)), hit_rate=float(np.mean(H)),
                coverage=len(rec_items) / S.shape[1], users=len(P))


def rating_eval(model, target):
    p = model.predict(target)
    return dict(rmse=rmse(target.rating, p), mae=mae(target.rating, p))
