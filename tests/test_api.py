import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, pytest
from api.app import create_app
from src.serving import Service, METHODS
from src import audit

@pytest.fixture(scope='module')
def svc():
    return Service.load('.')

@pytest.fixture(scope='module')
def client(svc):
    return create_app(svc).test_client()

@pytest.fixture(scope='module')
def uid(svc):
    return svc.user_ids[0]

def err(r, status, code):
    assert r.status_code == status, r.get_data(as_text=True)
    j = r.get_json(); assert set(j) == {'error'} and j['error']['code'] == code and j['error']['message']

# ---------------------------------------------------------------- normal cases
def test_health(client):
    j = client.get('/api/health').get_json()
    assert j['status'] == 'ok' and j['users'] == 309 and j['movies'] == 126 and 'SYNTHETIC' in j['data_note']

def test_recommend_default(client, svc, uid):
    j = client.get(f'/api/recommend?user_id={uid}').get_json()
    recs = j['recommendations']; seen = set(svc.history_df[svc.history_df.user_id == uid].movie_id)
    assert j['known_user'] and j['source'] == 'personalised' and j['method'] == 'deployed' and j['k'] == 10 and len(recs) == 10
    assert len({r['movie_id'] for r in recs}) == 10 and not {r['movie_id'] for r in recs} & seen
    assert [r['rank'] for r in recs] == list(range(1, 11)) and all(1 <= r['predicted_rating'] <= 5 for r in recs)

def test_all_methods(client, uid):
    for m in METHODS:
        j = client.get(f'/api/recommend?user_id={uid}&k=5&method={m}').get_json()
        assert len(j['recommendations']) == 5 and j['method'] == m
        assert (j['recommendations'][0]['predicted_rating'] is None) == (m == 'popularity')

def test_deterministic(client, uid):
    a = client.get(f'/api/recommend?user_id={uid}&k=7').get_json(); b = client.get(f'/api/recommend?user_id={uid}&k=7').get_json()
    assert a == b

def test_cold_start_fallback(client):
    j = client.get('/api/recommend?user_id=99999&k=8').get_json()
    assert not j['known_user'] and j['source'] == 'popularity_fallback' and j['diversify'] == 'safe' and len(j['recommendations']) == 8
    assert j['notes'] and all(r['predicted_rating'] is None for r in j['recommendations'])
    plain = client.get('/api/recommend?user_id=99999&k=8&diversify=none').get_json()
    assert plain['diversify'] == 'none' and plain['diversity']['combined'] <= j['diversity']['combined']

def test_safe_mode_matches_stage10_batch_procedure(client, svc):
    """API 'safe' must reproduce exactly the re-ranker evaluated in Stage 10 (gamma = 0)."""
    ref = svc.ref
    batch = audit.diversify(ref._S, ref.seen, svc.gcode, svc.lcode, 0.4, 0.0, 50, 10)
    for u in svc.user_ids[:25]:
        got = [r['movie_id'] for r in client.get(f'/api/recommend?user_id={u}&k=10&diversify=safe').get_json()['recommendations']]
        assert got == list(ref.item_ids[batch[ref.u2i[u]]])

def test_diversify_raises_variety_on_average(client, svc):
    plain, safe, div = [], [], []
    for u in svc.user_ids[:40]:
        for mode, bucket in (('none', plain), ('safe', safe), ('diverse', div)):
            bucket.append(client.get(f'/api/recommend?user_id={u}&k=10&diversify={mode}').get_json()['diversity']['combined'])
    assert np.mean(safe) > np.mean(plain) and np.mean(div) > np.mean(plain)

def test_similar(client, svc):
    a = client.get('/api/similar?movie=Inception&n=3').get_json(); b = client.get('/api/similar?movie=%20inception%20&n=3').get_json()
    mid = svc.movie_id('Inception'); c = client.get(f'/api/similar?movie={mid}&n=3').get_json()
    assert a == b == c and len(a['similar']) == 3 and a['movie_id'] == mid and mid not in {x['movie_id'] for x in a['similar']}

def test_history(client, svc, uid):
    j = client.get(f'/api/user/{uid}/history?limit=3').get_json()
    assert len(j['ratings']) == 3 and j['n_ratings'] == (svc.history_df.user_id == uid).sum()
    ts = [r['timestamp'] for r in client.get(f'/api/user/{uid}/history').get_json()['ratings']]
    assert ts == sorted(ts, reverse=True)

def test_explain_adds_up(client, uid):
    j = client.get(f'/api/explain?user_id={uid}&movie=Pink').get_json(); p = j['prediction']
    assert abs(p['predicted'] - min(5, max(1, p['global_mean'] + p['user_bias'] + p['movie_bias'] + p['cf_adjustment']))) < 0.01

# ---------------------------------------------------------------- boundary cases
def test_k_boundaries(client, uid):
    one = client.get(f'/api/recommend?user_id={uid}&k=1').get_json()
    assert len(one['recommendations']) == 1 and one['diversity'] is None
    for mode in ('none', 'safe', 'diverse'):
        j = client.get(f'/api/recommend?user_id={uid}&k=50&diversify={mode}').get_json()
        assert len(j['recommendations']) == 50 and len({r['movie_id'] for r in j['recommendations']}) == 50
    err(client.get(f'/api/recommend?user_id={uid}&k=0'), 400, 'out_of_range')
    err(client.get(f'/api/recommend?user_id={uid}&k=51'), 400, 'out_of_range')

def test_n_and_limit_boundaries(client, uid):
    assert len(client.get('/api/similar?movie=Inception&n=20').get_json()['similar']) == 20
    err(client.get('/api/similar?movie=Inception&n=21'), 400, 'out_of_range'); err(client.get('/api/similar?movie=Inception&n=0'), 400, 'out_of_range')
    err(client.get(f'/api/user/{uid}/history?limit=201'), 400, 'out_of_range')

# ---------------------------------------------------------------- invalid input
@pytest.mark.parametrize('qs,status,code', [
    ('', 400, 'missing_parameter'), ('user_id=', 400, 'missing_parameter'), ('user_id=abc', 400, 'invalid_parameter'), ('user_id=1.5', 400, 'invalid_parameter'),
    ('user_id=1e3', 400, 'invalid_parameter'), ('user_id=0', 400, 'out_of_range'), ('user_id=-5', 400, 'out_of_range'), ('user_id=10000000000', 400, 'out_of_range'),
    ('user_id=1&k=abc', 400, 'invalid_parameter'), ('user_id=1&k=2.5', 400, 'invalid_parameter'),
    ('user_id=1&method=nope', 400, 'invalid_parameter'), ('user_id=1&diversify=max', 400, 'invalid_parameter'),
    ("user_id=1;DROP TABLE users", 400, 'invalid_parameter'), ('user_id=<script>', 400, 'invalid_parameter')])
def test_recommend_invalid(client, qs, status, code):
    err(client.get('/api/recommend?' + qs), status, code)

def test_other_endpoints_invalid(client, uid):
    err(client.get('/api/similar'), 400, 'missing_parameter'); err(client.get('/api/similar?movie=Not+A+Movie'), 404, 'movie_not_found')
    err(client.get('/api/user/abc/history'), 400, 'invalid_parameter'); err(client.get('/api/user/99999/history'), 404, 'user_not_found')
    err(client.get('/api/explain?user_id=99999&movie=Pink'), 404, 'not_found'); err(client.get(f'/api/explain?user_id={uid}&movie=Nope'), 404, 'not_found')
    err(client.get(f'/api/explain?user_id={uid}'), 400, 'missing_parameter')
    err(client.get('/api/nope'), 404, 'not_found'); err(client.post('/api/health'), 405, 'method_not_allowed')

def test_error_messages_are_friendly(client):
    j = client.get('/api/recommend?user_id=abc').get_json()['error']
    assert 'whole number' in j['message'] and 'Traceback' not in str(j)
