"""User-based collaborative filtering (Stage 9).

  1. Baseline b_ui = mu + b_u + b_i (damped); residuals e_ui = r_ui - b_ui (a stronger form of mean-centring).
  2. User-user similarity over co-rated movies, either
       'pearson': Pearson correlation with means taken over the co-rated movies, or
       'cosine' : cosine of ratings centred on each user's overall mean (co-rated movies only).
     Both are multiplied by shrinkage n_uv / (n_uv + shrink) so pairs with few common movies count less.
  3. Prediction for (u, i) = b_ui + sum_v s_uv e_vi / (sum_v s_uv + damp) over the k most similar users v who rated i (positive s only).
"""
import numpy as np

from src.baselines import Base, damped_biases


class UserCF(Base):
    ranks = True

    def __init__(self, k=20, shrink=10.0, sim='pearson', damp=1.0, lam_i=5.0, lam_u=10.0, meta=None):
        assert sim in ('pearson', 'cosine')
        self.k, self.shrink, self.sim_type, self.damp, self.lam_i, self.lam_u = k, shrink, sim, damp, lam_i, lam_u
        self.meta = meta

    def _fit(self):
        self.bu, self.bi = damped_biases(self.mu, self.ui, self.ii, self.r, self.nU, self.nI, self.n_user, self.n_item, self.lam_i, self.lam_u)
        B = np.zeros((self.nU, self.nI)); B[self.ui, self.ii] = 1
        R = np.zeros((self.nU, self.nI)); R[self.ui, self.ii] = self.r
        self.B = B
        self.E = np.zeros((self.nU, self.nI)); self.E[self.ui, self.ii] = self.r - (self.mu + self.bu[self.ui] + self.bi[self.ii])
        n = B @ B.T
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.sim_type == 'pearson':
                Su, Q, RR = R @ B.T, (R ** 2) @ B.T, R @ R.T
                numer = RR - Su * Su.T / n
                var_u = Q - Su ** 2 / n
                den = np.sqrt(np.maximum(var_u, 0) * np.maximum(var_u.T, 0))
                ok = (n >= 2) & (den > 1e-9)
            else:
                C = (R - self.user_mean[:, None]) * B
                numer = C @ C.T
                A = (C ** 2) @ B.T
                den = np.sqrt(A * A.T)
                ok = den > 1e-9
            sim = np.where(ok, numer / den, 0.0)
        sim = np.clip(sim, -1, 1) * np.divide(n, n + self.shrink, out=np.zeros_like(n), where=(n + self.shrink) > 0)
        np.fill_diagonal(sim, 0.0)
        self.sim, self.cnt = sim, n

    def score_matrix(self):
        S = self.mu + self.bu[:, None] + self.bi[None, :]
        for u in range(self.nU):
            s = np.maximum(self.sim[u], 0.0)
            W = s[:, None] * self.B                         # weight of each user as a rater of each movie
            if self.k < self.nU:
                thr = np.partition(W, -self.k, axis=0)[-self.k][None, :]
                W = np.where(W >= thr, W, 0.0)              # keep the k most similar raters per movie
            num, den = (W * self.E).sum(0), W.sum(0) + self.damp
            S[u] += num / den
        return np.clip(S, 1, 5)
