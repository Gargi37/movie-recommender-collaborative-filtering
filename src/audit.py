"""Stage 10 toolkit: ranking metrics per user, diversity/exposure metrics, paired bootstrap, MMR + exposure re-ranker."""
import numpy as np
import pandas as pd
from scipy import stats

from src.data import RELEVANT


# ---------------------------------------------------------------- ranking
def ranked_lists(S, seen, n=50):
    """Top-n item indices per user (score desc, ties by item index), excluding movies already in that user's training data."""
    S2 = np.where(seen, -np.inf, S)
    ar = np.arange(S.shape[1])
    return np.array([np.lexsort((ar, -row))[:n] for row in S2])


def rank_metrics(lists, model, target, K=10, relevant=RELEVANT):
    """Per-user precision/recall/NDCG/hit at K for users with >= 1 relevant target movie. Binary relevance (rating >= relevant)."""
    rel = target[target.rating >= relevant]
    disc = 1 / np.log2(np.arange(2, K + 2))
    rows = []
    for u, g in rel.groupby('user_id'):
        if u not in model.u2i:
            rows.append((u, 0.0, 0.0, 0.0, 0.0)); continue
        relset = {model.i2i[m] for m in g.movie_id if m in model.i2i}
        hit = np.array([t in relset for t in lists[model.u2i[u], :K]])
        h, n_rel = int(hit.sum()), len(g)
        rows.append((u, h / K, h / n_rel, (disc * hit).sum() / disc[:min(K, n_rel)].sum(), float(h > 0)))
    return pd.DataFrame(rows, columns=['user_id', 'p', 'r', 'ndcg', 'hit']).set_index('user_id')


# ---------------------------------------------------------------- diversity / exposure
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else 0.0


def ild(codes, L):
    """Intra-list diversity: share of item pairs in a list that differ in `codes`, averaged over lists."""
    c = codes[L]; K = L.shape[1]
    return float(((c[:, :, None] != c[:, None, :]).sum((1, 2)) / (K * (K - 1))).mean())


def diversity(lists, model, gcode, lcode, K=10, tail_share=0.8, user_top_genres=None, hits=None):
    L = lists[:, :K]
    counts = np.bincount(L.ravel(), minlength=model.nI)
    order = np.argsort(-model.n_item, kind='stable'); head = np.zeros(model.nI, bool); head[order[:int(round(model.nI * (1 - tail_share)))]] = True
    p = model.n_item / model.n_item.sum()
    out = dict(ild_genre=ild(gcode, L), ild_language=ild(lcode, L), coverage=float((counts > 0).mean()), gini_exposure=gini(counts),
               tail_share_of_recs=float(counts[~head].sum() / counts.sum()), tail_items_recommended=int((counts[~head] > 0).sum()),
               tail_items_total=int((~head).sum()), novelty_bits=float(-np.log2(p[L]).mean()),
               pop_exposure_spearman=float(stats.spearmanr(model.n_item, counts)[0]) if counts.std() > 0 else float('nan'))
    if user_top_genres is not None:
        unexp = ~np.array([[gcode[i] in user_top_genres[u] for i in L[u]] for u in range(len(L))])
        out['unexpected_share'] = float(unexp.mean())
        if hits is not None:
            out['serendipity_per_list'] = float((unexp & hits[:, :K]).sum(1).mean())
    return out


def share_by_code(lists, codes, n_codes, K=10):
    c = np.bincount(codes[lists[:, :K]].ravel(), minlength=n_codes).astype(float)
    return c / c.sum()


# ---------------------------------------------------------------- paired bootstrap over users
def boot_weights(n_users, n_boot=2000, seed=42):
    return np.random.default_rng(seed).multinomial(n_users, np.full(n_users, 1 / n_users), size=n_boot)


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def per_user_sse(users, y, p, all_users):
    d = pd.DataFrame({'u': users, 'se': (np.asarray(y) - np.asarray(p)) ** 2, 'n': 1}).groupby('u').sum().reindex(all_users).fillna(0)
    return d.se.to_numpy(), d.n.to_numpy()


def boot_rmse_diff(sse_a, sse_b, cnt, W):
    ra, rb = np.sqrt(W @ sse_a / (W @ cnt)), np.sqrt(W @ sse_b / (W @ cnt))
    return ra - rb, ra, rb


# ---------------------------------------------------------------- re-ranker
def meta_similarity(gcode, lcode):
    return 0.5 * (gcode[:, None] == gcode[None, :]) + 0.5 * (lcode[:, None] == lcode[None, :])


def diversify(S, seen, gcode, lcode, lam=1.0, gamma=0.0, N=50, K=10):
    """Greedy MMR with an exposure penalty on the top-N candidates of each user.
    score_i = lam * rel_i - (1 - lam) * max_sim(i, chosen) - gamma * exposure_i
    rel_i = min-max normalised model score inside the user's candidate list; max_sim uses genre+language overlap (0..1);
    exposure_i = (times item i was already recommended) / (users processed so far + 1). Users are processed in index order.
    lam=1, gamma=0 returns the plain top-K."""
    nU, nI = S.shape
    sim = meta_similarity(gcode, lcode)
    cands = ranked_lists(S, seen, N)
    counts = np.zeros(nI); out = np.zeros((nU, K), int)
    for done in range(nU):
        c = cands[done]; sc = S[done, c]
        rel = (sc - sc.min()) / (sc.max() - sc.min() + 1e-12)
        expo = counts[c] / (done + 1)
        maxsim = np.zeros(len(c)); taken = np.zeros(len(c), bool)
        for k in range(K):
            val = lam * rel - (1 - lam) * maxsim - gamma * expo
            val[taken] = -np.inf
            j = int(np.argmax(val)); taken[j] = True
            out[done, k] = c[j]; counts[c[j]] += 1
            maxsim = np.maximum(maxsim, sim[c, c[j]])
    return out
