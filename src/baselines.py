"""Baselines (Stage 8). Every model exposes: fit(train), predict(df) -> ratings, score_matrix() -> users x items.
Index space (u2i, i2i) is fitted on the training data only. Cold-start policy (docs/split_report.md):
 unseen user -> movie mean; unseen movie -> user mean; both unseen -> global mean."""
import numpy as np
import pandas as pd

from src.data import build_index


def damped_biases(mu, ui, ii, r, nU, nI, n_user, n_item, lam_i=5.0, lam_u=10.0):
    """Item and user biases shrunk towards 0 by damping constants (fitted on training ratings only)."""
    res = r - mu
    bi = np.bincount(ii, res, nI) / (n_item + lam_i)
    bu = np.bincount(ui, res - bi[ii], nU) / (n_user + lam_u)
    return bu, bi


class Base:
    meta = None       # optional catalogue DataFrame for readable recommendations
    rates = True      # produces rating predictions
    ranks = False     # produces a meaningful top-N ranking

    def fit(self, train):
        self.u2i, self.i2i = build_index(train)
        self.nU, self.nI = len(self.u2i), len(self.i2i)
        self.mu = float(train.rating.mean())
        self.ui = train.user_id.map(self.u2i).to_numpy()
        self.ii = train.movie_id.map(self.i2i).to_numpy()
        self.r = train.rating.to_numpy(float)
        cu, ci = np.bincount(self.ui, minlength=self.nU), np.bincount(self.ii, minlength=self.nI)
        self.n_user, self.n_item = cu, ci
        self.user_ids, self.item_ids = np.array(list(self.u2i)), np.array(list(self.i2i))
        self.seen = np.zeros((self.nU, self.nI), bool); self.seen[self.ui, self.ii] = True
        self.user_mean = np.bincount(self.ui, self.r, self.nU) / np.maximum(cu, 1)
        self.item_mean = np.bincount(self.ii, self.r, self.nI) / np.maximum(ci, 1)
        self._fit()
        self._S = self.score_matrix()
        return self

    def _fit(self):
        pass

    def score_matrix(self):
        raise NotImplementedError

    def predict(self, df):
        """Predicted ratings clipped to [1, 5], with the cold-start policy applied."""
        if not self.rates:
            raise TypeError(f'{type(self).__name__} is a ranking-only model')
        u = df.user_id.map(self.u2i); i = df.movie_id.map(self.i2i)
        ku, ki = u.notna().to_numpy(), i.notna().to_numpy()
        uu, ii = u.fillna(0).astype(int).to_numpy(), i.fillna(0).astype(int).to_numpy()
        p = self._S[uu, ii].copy()
        p[ku & ~ki] = self.user_mean[uu[ku & ~ki]]
        p[~ku & ki] = self.item_mean[ii[~ku & ki]]
        p[~ku & ~ki] = self.mu
        return np.clip(p, 1, 5)


    def _info(self, idx):
        d = pd.DataFrame({'movie_id': self.item_ids[idx]})
        return d if self.meta is None else d.merge(self.meta, on='movie_id', how='left')

    def recommend(self, user_id, k=10, exclude_seen=True):
        """Top-k movies for a known user ranked by score (predicted rating for rating models)."""
        if user_id not in self.u2i:
            raise KeyError(f'unknown user {user_id}: use the popularity fallback')
        u = self.u2i[user_id]
        row = self._S[u].copy()
        if exclude_seen:
            row[self.seen[u]] = -np.inf
        top = np.lexsort((np.arange(self.nI), -row))[:k]
        out = self._info(top); out.insert(1, 'score', row[top].round(3))
        return out


class GlobalMean(Base):
    def score_matrix(self):
        return np.full((self.nU, self.nI), self.mu)


class UserMean(Base):
    def score_matrix(self):
        return np.repeat(self.user_mean[:, None], self.nI, axis=1)


class ItemMean(Base):
    def score_matrix(self):
        return np.repeat(self.item_mean[None, :], self.nU, axis=0)


class BiasBaseline(Base):
    """mu + b_item + b_user with damping (shrinks biases of thinly rated items/users towards 0)."""
    def __init__(self, lam_i=5.0, lam_u=10.0):
        self.lam_i, self.lam_u = lam_i, lam_u

    def _fit(self):
        self.bu, self.bi = damped_biases(self.mu, self.ui, self.ii, self.r, self.nU, self.nI, self.n_user, self.n_item, self.lam_i, self.lam_u)

    def score_matrix(self):
        return self.mu + self.bu[:, None] + self.bi[None, :]


class RandomRec(Base):
    rates, ranks = False, True
    def __init__(self, seed=42):
        self.seed = seed
    def score_matrix(self):
        return np.random.default_rng(self.seed).random((self.nU, self.nI))


class PopularityRec(Base):
    """Most-rated movies first (same list for every user, minus what they have seen)."""
    rates, ranks = False, True
    def score_matrix(self):
        return np.repeat(self.n_item[None, :].astype(float), self.nU, axis=0)


class TopRatedRec(Base):
    """Highest average rating first, shrunk towards the global mean so thin movies do not top the list."""
    rates, ranks = False, True
    def __init__(self, m=10.0):
        self.m = m
    def score_matrix(self):
        shrunk = (self.item_mean * self.n_item + self.m * self.mu) / (self.n_item + self.m)
        return np.repeat(shrunk[None, :], self.nU, axis=0)
