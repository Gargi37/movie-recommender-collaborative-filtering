"""Shared data helpers: loading, per-user chronological split, index mappings, sparse matrices."""
import numpy as np
import pandas as pd
from scipy import sparse

RELEVANT = 4.0          # rating >= 4.0 counts as relevant for ranking metrics
TEST_FRAC, VAL_FRAC = 0.20, 0.20
MODEL_COLS = ['user_id', 'movie_id', 'rating', 'timestamp']   # the ONLY columns models may use (is_original excluded)


def load_clean(path='data/processed/dataset.csv'):
    return pd.read_csv(path, parse_dates=['timestamp'])


def per_user_split(df, frac):
    """Per user, the latest round(frac*n) ratings (at least 1) go to the second part. Returns (early, late).
    Sorted by (timestamp, movie_id) so ties are deterministic."""
    d = df.sort_values(['user_id', 'timestamp', 'movie_id'], kind='stable').copy()
    pos = d.groupby('user_id').cumcount().to_numpy()
    n = d.groupby('user_id').user_id.transform('size').to_numpy()
    k = np.maximum(1, np.floor(frac * n + 0.5).astype(int))
    late = pos >= (n - k)
    return d[~late].reset_index(drop=True), d[late].reset_index(drop=True)


def make_splits(df):
    train_full, test = per_user_split(df, TEST_FRAC)
    train_core, val = per_user_split(train_full, VAL_FRAC)
    return train_core, val, train_full, test


def build_index(train):
    """Index mappings fitted on TRAINING data only."""
    users = np.sort(train.user_id.unique()); items = np.sort(train.movie_id.unique())
    return {u: i for i, u in enumerate(users)}, {m: i for i, m in enumerate(items)}


def to_csr(df, u2i, i2i):
    d = df[df.user_id.isin(u2i) & df.movie_id.isin(i2i)]
    return sparse.csr_matrix((d.rating.to_numpy(float), (d.user_id.map(u2i).to_numpy(), d.movie_id.map(i2i).to_numpy())),
                             shape=(len(u2i), len(i2i)))


def mean_centre(R):
    """Subtract each user's mean (over rated cells only) from a CSR rating matrix. Returns (centred CSR, user_means)."""
    R = R.tocsr().astype(float); cnt = np.diff(R.indptr); sums = np.asarray(R.sum(1)).ravel()
    means = np.divide(sums, cnt, out=np.zeros_like(sums), where=cnt > 0)
    C = R.copy(); C.data = C.data - np.repeat(means, cnt)
    return C, means
