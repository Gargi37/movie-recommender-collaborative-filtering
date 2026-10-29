import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from src.data import per_user_split, make_splits, load_clean, build_index, to_csr, mean_centre

def toy():
    rows = [(u, f'M{m}', 3.0, pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(d)))
            for u in (1, 2) for m, d in enumerate(np.random.default_rng(u).permutation(15))]
    return pd.DataFrame(rows, columns=['user_id', 'movie_id', 'rating', 'timestamp'])

def test_latest_goes_to_late_part_per_user():
    early, late = per_user_split(toy(), 0.2)
    for u in (1, 2):
        assert early[early.user_id == u].timestamp.max() <= late[late.user_id == u].timestamp.min()
        assert len(late[late.user_id == u]) == 3          # round(0.2*15)

def test_partition_and_deterministic():
    d = toy(); e1, l1 = per_user_split(d, 0.2); e2, l2 = per_user_split(d.sample(frac=1, random_state=1), 0.2)
    assert len(e1) + len(l1) == len(d) and l1.equals(l2)

def test_real_data_properties():
    df = load_clean(); tc, v, tf, te = make_splits(df)
    assert set(te.user_id) <= set(tf.user_id) and set(v.user_id) <= set(tc.user_id)      # every user in train
    assert len(tc) + len(v) + len(te) == len(df)
    assert te.groupby('user_id').size().min() >= 1

def test_matrix_and_centering():
    df = load_clean(); _, _, tf, _ = make_splits(df); u2i, i2i = build_index(tf); R = to_csr(tf, u2i, i2i)
    assert R.shape == (len(u2i), len(i2i)) and R.nnz == len(tf)
    C, means = mean_centre(R)
    assert abs(np.asarray(C.sum(1)).ravel()).max() < 1e-9      # each user's centred ratings sum to 0
