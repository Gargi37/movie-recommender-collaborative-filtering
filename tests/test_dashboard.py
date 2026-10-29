import sys; sys.path.insert(0, '.')
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app' / 'dashboard.py')

def test_dashboard_runs_and_responds_to_controls():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception and len(at.dataframe) >= 1
    at.selectbox(key='method').select('svd').run(); assert not at.exception
    at.selectbox(key='diversify').select('safe').run(); assert not at.exception
    at.slider(key='k').set_value(5).run(); assert not at.exception
    at.selectbox(key='mode').select('New user (cold start)').run(); assert not at.exception
    assert any('cold-start' in i.value.lower() for i in at.info)
    at.selectbox(key='mode').select('Existing user').run(); assert not at.exception
