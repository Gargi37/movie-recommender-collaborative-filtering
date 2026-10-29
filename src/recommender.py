"""Integrated recommender (Stage 9): data layer -> tuned CF model -> popularity blend -> recommend().
Unknown users (cold start) get a popularity list; known users get personalised results.
Persisted with joblib by src/finalize_stage9.py; the API (Stage 11) and the dashboard only talk to this class."""
import joblib
import numpy as np
import pandas as pd


class Recommender:
    def __init__(self, model, popularity, meta, history):
        self.model, self.popularity, self.meta, self.history_df = model, popularity, meta, history
        self.base = getattr(model, 'base', model)           # the CF model that predicts ratings

    @classmethod
    def load(cls, path='models/final_recommender.joblib'):
        d = joblib.load(path)
        return cls(d['model'], d['popularity'], d['meta'], d['history'])

    def save(self, path='models/final_recommender.joblib'):
        joblib.dump(dict(model=self.model, popularity=self.popularity, meta=self.meta, history=self.history_df), path)

    def knows_user(self, user_id):
        return user_id in self.model.u2i

    def recommend(self, user_id, k=10, exclude_seen=True):
        """Top-k unseen movies. Columns: movie_id, score, movie metadata, predicted_rating (None for cold users), source."""
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError('k must be a positive integer')
        k = min(int(k), self.model.nI)
        if self.knows_user(user_id):
            out = self.model.recommend(user_id, k, exclude_seen)
            u = self.model.u2i[user_id]
            out['predicted_rating'] = [round(float(self.base._S[u, self.model.i2i[m]]), 2) for m in out.movie_id]
            out['source'] = 'personalised'
        else:                                               # cold start: most-rated movies
            idx = np.lexsort((np.arange(self.popularity.nI), -self.popularity.n_item))[:k]
            out = self.popularity._info(idx); out.insert(1, 'score', self.popularity.n_item[idx].astype(float))
            out['predicted_rating'] = None
            out['source'] = 'popularity_fallback'
        return out

    def predict_rating(self, user_id, movie_id):
        """Predicted rating with the documented cold-start policy (unseen user -> movie mean, unseen movie -> user mean)."""
        return float(self.base.predict(pd.DataFrame({'user_id': [user_id], 'movie_id': [movie_id]}))[0])

    def similar_movies(self, movie_id, n=5):
        if not hasattr(self.base, 'similar_items'):
            raise TypeError('the deployed base model has no item similarities')
        return self.base.similar_items(movie_id, n)

    def explain(self, user_id, movie_id, n=3):
        """Why this predicted rating: baseline parts + the movies the user rated that pulled it up or down (item-based CF only)."""
        if not hasattr(self.base, 'explain'):
            raise TypeError('the deployed base model cannot explain predictions')
        if user_id not in self.base.u2i or movie_id not in self.base.i2i:
            raise KeyError('unknown user or movie: no personalised explanation (cold-start fallback applies)')
        return self.base.explain(user_id, movie_id, n)

    def history(self, user_id):
        h = self.history_df[self.history_df.user_id == user_id].sort_values('timestamp')
        return h[['movie_id', 'movie_title', 'genre', 'language', 'rating', 'timestamp']]
