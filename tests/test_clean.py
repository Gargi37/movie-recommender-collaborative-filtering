import sys; sys.path.insert(0, '.')
import pandas as pd
from src.clean import clean_ratings

CAT = pd.DataFrame({'movie_id': ['M001', 'M002'], 'movie_title': ['A', 'B'], 'genre': ['Action', 'Drama'],
                    'language': ['Hindi', 'Tamil'], 'release_year': [2010, 2024]})

def row(u, m, r, t, g='Action', l='Hindi', y=2010, title='A', orig=False):
    return dict(user_id=u, movie_title=title, genre=g, language=l, release_year=y, rating=r, timestamp=t, movie_id=m, is_original=orig)

def run(rows):
    return clean_ratings(pd.DataFrame(rows), CAT)[0]

def test_valid_row_survives():
    assert len(run([row(1, 'M001', 4.0, '2024-05-01 10:00:00')])) == 1

def test_text_standardised():
    out = run([row(1, 'M001', 4.0, '2024-05-01 10:00:00', g=' action ', l='HINDI')])
    assert out.genre[0] == 'Action' and out.language[0] == 'Hindi'

def test_duplicates_keep_latest():
    out = run([row(1, 'M001', 2.0, '2024-01-01 10:00:00'), row(1, 'M001', 5.0, '2024-06-01 10:00:00')])
    assert len(out) == 1 and out.rating[0] == 5.0

def test_bad_ratings_dropped():
    out = run([row(1, 'M001', 6.0, '2024-05-01 10:00:00'), row(2, 'M001', 0.0, '2024-05-01 10:00:00'), row(3, 'M001', 3.3, '2024-05-01 10:00:00')])
    assert len(out) == 0

def test_missing_and_bad_dates_dropped():
    out = run([row(1, 'M001', None, '2024-05-01 10:00:00'), row(2, 'M001', 4.0, 'not a date')])
    assert len(out) == 0

def test_rating_before_release_dropped():
    assert len(run([row(1, 'M002', 4.0, '2024-05-01 10:00:00', y=2024, title='B', g='Drama', l='Tamil')])) == 1
    assert len(run([row(1, 'M002', 4.0, '2023-05-01 10:00:00', y=2024, title='B', g='Drama', l='Tamil')])) == 0

def test_unknown_movie_dropped_and_attrs_from_catalogue():
    assert len(run([row(1, 'M999', 4.0, '2024-05-01 10:00:00')])) == 0
    out = run([row(1, 'M001', 4.0, '2024-05-01 10:00:00', g='Comedy', l='Tamil', y=2015)])
    assert out.genre[0] == 'Action' and out.language[0] == 'Hindi' and out.release_year[0] == 2010
