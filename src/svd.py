"""Matrix factorisation with the Surprise library (Stage 9): r_hat = mu + b_u + b_i + p_u . q_i, trained by SGD.
Wrapped so it shares the Base interface. If scikit-surprise cannot be installed, see docs/README (NumPy fallback)."""
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD

from src.baselines import Base


class SVDMF(Base):
    ranks = True

    def __init__(self, n_factors=20, lr=0.005, reg=0.02, epochs=20, seed=42, meta=None):
        self.n_factors, self.lr, self.reg, self.epochs, self.seed = n_factors, lr, reg, epochs, seed
        self.meta = meta

    def _fit(self):
        df = pd.DataFrame({'u': self.ui, 'i': self.ii, 'r': self.r})
        ts = Dataset.load_from_df(df, Reader(rating_scale=(1, 5))).build_full_trainset()
        algo = SVD(n_factors=self.n_factors, n_epochs=self.epochs, lr_all=self.lr, reg_all=self.reg, random_state=self.seed)
        algo.fit(ts)
        # map Surprise's inner ids back to our index space; keep only plain arrays (small, picklable)
        iu = np.array([ts.to_inner_uid(u) if ts.knows_user(u) else -1 for u in range(self.nU)])
        ii = np.array([ts.to_inner_iid(i) if ts.knows_item(i) else -1 for i in range(self.nI)])
        assert (iu >= 0).all() and (ii >= 0).all()
        self.bu, self.bi = algo.bu[iu], algo.bi[ii]
        self.P, self.Q = algo.pu[iu], algo.qi[ii]
        self.mu_s = ts.global_mean

    def score_matrix(self):
        return np.clip(self.mu_s + self.bu[:, None] + self.bi[None, :] + self.P @ self.Q.T, 1, 5)
