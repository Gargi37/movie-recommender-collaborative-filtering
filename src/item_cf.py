"""Item-based collaborative filtering (Stage 8 first working module).

Method
  1. Baseline b_ui = mu + b_u + b_i (damped), fitted on training data only.
  2. Residuals e_ui = r_ui - b_ui for every training rating.
  3. Item-item similarity = cosine of residual columns over co-rating users (adjusted-cosine style),
     multiplied by shrinkage n_ij / (n_ij + shrink) so pairs with few common raters count less.
  4. Prediction for (u, i) = b_ui + sum_j s_ij e_uj / sum_j s_ij over the k most similar movies j that u rated (positive s only).
     No usable neighbour -> falls back to the baseline.
  5. Damping: the denominator is sum_j s_ij + damp. Without it (damp=0) the size of the similarities cancels out and one weak
     neighbour moves the prediction as much as a strong one (found in Stage 8, experiment E08 vs E09). damp -> infinity gives the baseline.
Defaults (k=20, shrink=10) are untuned; they are tuned on the validation set in Stage 9."""
import numpy as np
import pandas as pd

from src.baselines import Base, damped_biases


class ItemCF(Base):
    ranks = True

    def __init__(self, k=20, shrink=10.0, lam_i=5.0, lam_u=10.0, damp=0.0, meta=None):
        self.k, self.shrink, self.lam_i, self.lam_u, self.damp = k, shrink, lam_i, lam_u, damp
        self.meta = meta                      # optional catalogue DataFrame (movie_id, movie_title, genre, language, release_year)

    def _fit(self):
        self.bu, self.bi = damped_biases(self.mu, self.ui, self.ii, self.r, self.nU, self.nI, self.n_user, self.n_item, self.lam_i, self.lam_u)
        self.B = np.zeros((self.nU, self.nI)); self.B[self.ui, self.ii] = 1
        base = self.mu + self.bu[self.ui] + self.bi[self.ii]
        self.E = np.zeros((self.nU, self.nI)); self.E[self.ui, self.ii] = self.r - base
        N = self.E.T @ self.E                               # sum over co-raters of e_ui * e_uj
        A = (self.E ** 2).T @ self.B                        # A_ij = squared norm of item i over raters of j
        cnt = self.B.T @ self.B                             # number of co-raters
        den = np.sqrt(A * A.T)
        cos = np.divide(N, den, out=np.zeros_like(N), where=den > 0)
        self.sim = cos * np.divide(cnt, cnt + self.shrink, out=np.zeros_like(cnt), where=(cnt + self.shrink) > 0)
        np.fill_diagonal(self.sim, 0.0)
        self.cnt = cnt

    def score_matrix(self):
        S = self.mu + self.bu[:, None] + self.bi[None, :]
        for u in range(self.nU):
            J = np.flatnonzero(self.B[u])
            if J.size == 0:
                continue
            s = np.maximum(self.sim[:, J], 0.0)             # items x rated-items, positive similarities only
            if J.size > self.k:
                thr = np.partition(s, -self.k, axis=1)[:, -self.k][:, None]
                s = np.where(s >= thr, s, 0.0)              # keep the k most similar
            num, den = s @ self.E[u, J], s.sum(1) + self.damp      # damp pulls weakly supported predictions back to the baseline
            S[u] += np.divide(num, den, out=np.zeros(self.nI), where=den > 0)
        return np.clip(S, 1, 5)

    # ---------- demo / API helpers (recommend() is inherited from Base) ----------
    def similar_items(self, movie_id, n=5):
        if movie_id not in self.i2i:
            raise KeyError(f'unknown movie {movie_id}')
        i = self.i2i[movie_id]
        top = np.lexsort((np.arange(self.nI), -self.sim[i]))[:n]
        out = self._info(top); out.insert(1, 'similarity', self.sim[i, top].round(3))
        out.insert(2, 'co_raters', self.cnt[i, top].astype(int))
        return out

    def explain(self, user_id, movie_id, n=3):
        """Break a predicted rating into baseline parts and the CF adjustment, with the user's movies that contributed most."""
        u, i = self.u2i[user_id], self.i2i[movie_id]
        J = np.flatnonzero(self.B[u]); s = np.maximum(self.sim[i, J], 0.0)
        if J.size > self.k:
            keep = np.argsort(-s, kind='stable')[:self.k]; mask = np.zeros(J.size, bool); mask[keep] = True; s = np.where(mask, s, 0.0)
        den = s.sum() + self.damp
        contrib = s * self.E[u, J] / den
        order = np.argsort(-np.abs(contrib), kind='stable')[:n]
        nb = self._info(J[order]); nb.insert(1, 'similarity', s[order].round(3)); nb.insert(2, 'your_residual', self.E[u, J][order].round(2))
        nb.insert(3, 'contribution', contrib[order].round(3))
        parts = dict(global_mean=self.mu, user_bias=float(self.bu[u]), movie_bias=float(self.bi[i]), cf_adjustment=float(contrib.sum()))
        parts['predicted'] = float(np.clip(sum(parts.values()), 1, 5))
        return parts, nb
