"""Blend of a CF score with popularity for ranking: score = z(CF prediction within user) + alpha * z(log popularity).
Ratings predictions (predict) are those of the base CF model, unchanged. alpha = 0 reproduces the base model's ranking."""
import numpy as np
from src.baselines import Base


class BlendRec(Base):
    ranks = True

    def __init__(self, base, alpha=1.0, meta=None):
        self.base, self.alpha, self.meta = base, alpha, meta

    def fit(self, train):
        self.base.fit(train)
        return super().fit(train)

    def score_matrix(self):
        S = self.base._S
        z = (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-9)
        pop = np.log1p(self.n_item); pop = (pop - pop.mean()) / pop.std()
        return z + self.alpha * pop[None, :]

    def predict(self, df):
        return self.base.predict(df)
