"""Serving layer shared by the Flask API and the Streamlit dashboard, so both always return the same results.

Loads the models saved in Stage 9 (models/*.joblib, fitted on train_full) and exposes recommend / similar / history / explain.
Re-ranking: 'safe' = MMR with lambda 0.4, gamma 0 (exactly the procedure evaluated in Stage 10);
'diverse' = lambda 0.6, gamma 0.5 with a STATIC exposure estimate (a single-user approximation of the sequential batch procedure)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.audit import diversify as batch_diversify, meta_similarity
from src.generate_data import GENRES, LANGS

METHODS = {'deployed': 'item_cf_blend', 'item_cf': 'item_cf', 'user_cf': 'user_cf', 'svd': 'svd', 'svd_blend': 'svd_blend', 'popularity': 'popularity'}
METHOD_LABELS = {'deployed': 'Item CF + popularity (deployed)', 'item_cf': 'Item-based CF', 'user_cf': 'User-based CF', 'svd': 'SVD (matrix factorisation)',
                 'svd_blend': 'SVD + popularity', 'popularity': 'Most popular'}
DIVERSIFY_MODES = ['none', 'safe', 'diverse']
COLD_ID = 999_999_999                     # sentinel the dashboard uses for "new user"
DATA_NOTE = 'Demo on SYNTHETIC data (extended from the provided file); results are not findings about real viewers.'


class Service:
    def __init__(self, models, cat, history, mitigation):
        self.models, self.cat, self.history_df = models, cat.set_index('movie_id'), history
        ref = models['item_cf_blend']
        self.ref, self.nI = ref, ref.nI
        self.item_ids = ref.item_ids
        self.item_meta = self.cat.loc[self.item_ids]
        self.gcode = np.array([GENRES.index(g) for g in self.item_meta.genre]); self.lcode = np.array([LANGS.index(l) for l in self.item_meta.language])
        self.sim = meta_similarity(self.gcode, self.lcode)
        self.title_to_id = {t.lower(): m for m, t in self.cat.movie_title.items()}
        self.user_ids = sorted(int(u) for u in ref.u2i)
        self.modes = {'none': None, 'safe': (mitigation['safe']['lam'], mitigation['safe']['gamma']),
                      'diverse': (mitigation['diverse']['lam'], mitigation['diverse']['gamma'])}
        lists = batch_diversify(ref._S, ref.seen, self.gcode, self.lcode, *self.modes['diverse'], 50, 10)      # static exposure estimate
        self.expo = np.bincount(lists.ravel(), minlength=self.nI) / ref.nU

    @classmethod
    def load(cls, root='.'):
        root = Path(root)
        models = {k: joblib.load(root / f'models/{k}.joblib') for k in set(METHODS.values())}
        cat = pd.read_csv(root / 'data/raw/movie_catalogue.csv')
        history = pd.read_csv(root / 'data/processed/split/train_full.csv')
        return cls(models, cat, history, json.load(open(root / 'results/mitigation_choice.json')))

    # ------------------------------------------------------------------ helpers
    def knows_user(self, user_id):
        return user_id in self.ref.u2i

    def movie_id(self, movie):
        """Accept a movie_id (M001) or a title (case-insensitive). Returns movie_id or None."""
        if movie in self.cat.index:
            return movie
        return self.title_to_id.get(str(movie).strip().lower())

    def _rerank(self, S_row, seen_row, mode, k):
        lam, gamma = self.modes[mode]
        s = np.where(seen_row, -np.inf, S_row)
        order = np.lexsort((np.arange(len(s)), -s))
        n_avail = int(np.isfinite(s).sum())
        cand = order[:min(max(50, 2 * k), n_avail)]
        sc = S_row[cand]; rel = (sc - sc.min()) / (sc.max() - sc.min() + 1e-12)
        ex = self.expo[cand]
        maxsim = np.zeros(len(cand)); taken = np.zeros(len(cand), bool); out = []
        for _ in range(min(k, len(cand))):
            val = lam * rel - (1 - lam) * maxsim - gamma * ex
            val[taken] = -np.inf
            j = int(np.argmax(val)); taken[j] = True; out.append(int(cand[j]))
            maxsim = np.maximum(maxsim, self.sim[cand, cand[j]])
        return np.array(out)

    def _diversity(self, idx):
        if len(idx) < 2:
            return None
        def ild(codes):
            c = codes[idx]; return float((c[:, None] != c[None, :]).sum() / (len(c) * (len(c) - 1)))
        g, l = ild(self.gcode), ild(self.lcode)
        return dict(genre=round(g, 3), language=round(l, 3), combined=round((g + l) / 2, 3),
                    distinct_genres=int(len(set(self.gcode[idx]))), distinct_languages=int(len(set(self.lcode[idx]))))

    # ------------------------------------------------------------------ public API
    def recommend(self, user_id, k=10, method='deployed', diversify=None):
        """Top-k unseen movies. Unknown users get the most-rated movies, re-ranked for variety by default (cold-start fallback)."""
        known = self.knows_user(user_id); notes = []
        if known:
            model = self.models[METHODS[method]]; u = self.ref.u2i[user_id]
            S_row, seen_row = model._S[u], self.ref.seen[u]
            base = model.base if hasattr(model, 'base') else model
            rating_row = base._S[u] if method != 'popularity' else None
            source = 'personalised' if method != 'popularity' else 'popularity'
            mode = diversify or 'none'
        else:
            pop = self.models['popularity']
            S_row, seen_row, rating_row = pop._S[0].astype(float), np.zeros(self.nI, bool), None
            source, mode = 'popularity_fallback', diversify or 'safe'
            notes.append('Unknown user: showing the most-rated movies, re-ranked for genre/language variety (cold-start fallback). Method is ignored.')
        if method == 'popularity' and known:
            notes.append('Popularity ignores your history except to hide movies you already rated.')
        if self.modes[mode] is None:
            s = np.where(seen_row, -np.inf, S_row)
            idx = np.lexsort((np.arange(self.nI), -s))[:k]
        else:
            idx = self._rerank(S_row, seen_row, mode, k)
        recs = []
        for rank, i in enumerate(idx, 1):
            m = self.item_meta.iloc[i]
            recs.append(dict(rank=rank, movie_id=str(self.item_ids[i]), title=m.movie_title, genre=m.genre, language=m.language, release_year=int(m.release_year),
                             predicted_rating=None if rating_row is None else round(float(rating_row[i]), 2), score=round(float(S_row[i]), 3)))
        return dict(user_id=int(user_id), known_user=known, method=method if known else 'popularity_fallback', diversify=mode, k=int(k), source=source,
                    recommendations=recs, diversity=self._diversity(np.array(idx)), notes=notes, data_note=DATA_NOTE)

    def similar(self, movie, n=5):
        mid = self.movie_id(movie)
        if mid is None or mid not in self.models['item_cf'].i2i:
            return None
        df = self.models['item_cf'].similar_items(mid, n)
        rows = [dict(movie_id=r.movie_id, title=r.movie_title, genre=r.genre, language=r.language, release_year=int(r.release_year),
                     similarity=float(r.similarity), co_raters=int(r.co_raters)) for r in df.itertuples()]
        return dict(movie_id=mid, title=self.cat.loc[mid, 'movie_title'], similar=rows,
                    note='Similarities rest on few co-raters (typically 2-5 per pair) and are weak on this sparse synthetic data.', data_note=DATA_NOTE)

    def history(self, user_id, limit=50):
        if not self.knows_user(user_id):
            return None
        h = self.history_df[self.history_df.user_id == user_id].sort_values('timestamp', ascending=False)
        rows = [dict(movie_id=r.movie_id, title=r.movie_title, genre=r.genre, language=r.language, rating=float(r.rating), timestamp=str(r.timestamp))
                for r in h.head(limit).itertuples()]
        return dict(user_id=int(user_id), n_ratings=int(len(h)), mean_rating=round(float(h.rating.mean()), 2), ratings=rows,
                    note='Training ratings only (the model was fitted on these).', data_note=DATA_NOTE)

    def explain(self, user_id, movie):
        mid = self.movie_id(movie)
        base = self.models['item_cf']
        if not self.knows_user(user_id) or mid is None or mid not in base.i2i:
            return None
        parts, nb = base.explain(user_id, mid, 3)
        return dict(user_id=int(user_id), movie_id=mid, title=self.cat.loc[mid, 'movie_title'], seen_by_user=bool(base.seen[base.u2i[user_id], base.i2i[mid]]),
                    prediction={k: round(v, 3) for k, v in parts.items()},
                    influencing_movies=[dict(movie_id=r.movie_id, title=r.movie_title, similarity=float(r.similarity), your_residual=float(r.your_residual),
                                             contribution=float(r.contribution)) for r in nb.itertuples()],
                    note='Item-based CF explanation: prediction = global mean + user bias + movie bias + adjustment from similar movies you rated.', data_note=DATA_NOTE)

    def health(self):
        return dict(status='ok', users=len(self.user_ids), movies=int(self.nI), methods=list(METHODS), diversify_modes=DIVERSIFY_MODES,
                    deployed_method=METHOD_LABELS['deployed'], data_note=DATA_NOTE)
