"""Streamlit demo.  Run from the repo root:  streamlit run app/dashboard.py
Uses the same Service class as the REST API, so both always give the same recommendations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st

from src.serving import Service, METHODS, METHOD_LABELS, DIVERSIFY_MODES, COLD_ID

st.set_page_config(page_title='Personalized Movie Recommender', layout='wide')


@st.cache_resource
def load_service():
    return Service.load(Path(__file__).resolve().parents[1])


S = load_service()
DIV_LABEL = {
    'none': 'Off (most relevant only)',
    'safe': 'Safe (more variety, tiny accuracy cost)',
    'diverse': 'Diverse (most variety, more accuracy cost)'
}

st.title('Personalized Movie Recommender')

with st.sidebar:
    st.header('Controls')
    mode = st.selectbox('Viewer', ['Existing user', 'New user (cold start)'], key='mode')
    uid = st.selectbox('User ID', S.user_ids, key='user') if mode == 'Existing user' else COLD_ID
    method = st.selectbox('Method', list(METHODS), format_func=METHOD_LABELS.get, key='method', disabled=mode != 'Existing user')
    diversify = st.selectbox('Diversify results', DIVERSIFY_MODES, index=0 if mode == 'Existing user' else 1, format_func=DIV_LABEL.get, key='diversify')
    k = st.slider('Number of recommendations', 5, 20, 10, key='k')
    st.caption('"Diversify" re-ranks the top candidates so the list mixes genres and languages. Chosen on validation data in Stage 10.')

res = S.recommend(int(uid), k, method, diversify)
recs = pd.DataFrame(res['recommendations'])
hist = S.history(int(uid)) if res['known_user'] else None

c1, c2, c3, c4 = st.columns(4)
c1.metric('Ratings by this user', hist['n_ratings'] if hist else 0)
c2.metric('List diversity (0-1)', res['diversity']['combined'] if res['diversity'] else '-')
c3.metric('Distinct languages in list', res['diversity']['distinct_languages'] if res['diversity'] else '-')
c4.metric('Source', res['source'].replace('_', ' '))
for n in res['notes']:
    st.info(n)

tab_rec, tab_why, tab_hist, tab_sim, tab_eval, tab_about = st.tabs(['Recommendations', 'Why this movie?', 'History', 'Similar movies', 'How good is it?', 'About and ethics'])

with tab_rec:
    show = recs[['rank', 'title', 'genre', 'language', 'release_year', 'predicted_rating']].rename(
        columns={'title': 'Movie', 'genre': 'Genre', 'language': 'Language', 'release_year': 'Year', 'predicted_rating': 'Predicted rating'})
    st.dataframe(show, hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    cat_g = S.item_meta.genre.value_counts(normalize=True).mul(100).rename('Catalogue')
    cat_l = S.item_meta.language.value_counts(normalize=True).mul(100).rename('Catalogue')
    a.subheader('Genre mix of this list vs catalogue (%)')
    a.bar_chart(pd.concat([cat_g, recs.genre.value_counts(normalize=True).mul(100).rename('This list')], axis=1).fillna(0), x_label='Genre', y_label='Share (%)')
    b.subheader('Language mix of this list vs catalogue (%)')
    b.bar_chart(pd.concat([cat_l, recs.language.value_counts(normalize=True).mul(100).rename('This list')], axis=1).fillna(0), x_label='Language', y_label='Share (%)')

with tab_why:
    if not res['known_user']:
        st.write('New users have no history to explain, so they see the most-rated movies (re-ranked for variety).')
    else:
        title = st.selectbox('Pick a movie from the list', recs.title.tolist(), key='why_movie')
        e = S.explain(int(uid), title)
        if e is None:
            st.write('No explanation available for this movie.')
        else:
            p = e['prediction']
            st.write(f"Predicted rating **{p['predicted']:.2f}** = global mean {p['global_mean']:.2f} + user bias {p['user_bias']:+.2f} + movie bias {p['movie_bias']:+.2f} + similar-movie adjustment {p['cf_adjustment']:+.2f}")
            st.caption(e['note'])
            if e['influencing_movies']:
                st.dataframe(pd.DataFrame(e['influencing_movies'])[['title', 'similarity', 'your_residual', 'contribution']], hide_index=True)
            st.caption('Most of a prediction comes from the biases; similar-movie adjustments are small on this sparse data (see the evaluation tab).')

with tab_hist:
    if hist:
        st.write(f"{hist['n_ratings']} ratings, average {hist['mean_rating']}.")
        st.dataframe(pd.DataFrame(hist['ratings'])[['title', 'genre', 'language', 'rating', 'timestamp']], hide_index=True, use_container_width=True)
    else:
        st.write('New user: no history.')

with tab_sim:
    movie = st.selectbox('Movie', sorted(S.cat.movie_title), key='sim_movie')
    sim = S.similar(movie, 5)
    st.dataframe(pd.DataFrame(sim['similar'])[['title', 'genre', 'language', 'similarity', 'co_raters']], hide_index=True)
    st.caption(sim['note'])

with tab_eval:
    st.write('Final test-set results (Stage 10; the test set was opened once). Full details: docs/evaluation_report.md.')
    m = pd.read_csv(Path(__file__).resolve().parents[1] / 'results/test_metrics.csv')
    st.subheader('Rating prediction (lower RMSE is better)')
    st.dataframe(m[m.task == 'rating'][['model', 'RMSE', 'MAE']].round(4), hide_index=True)
    st.subheader('Top-10 ranking')
    st.dataframe(m[m.task == 'ranking'][['model', 'P@10', 'NDCG@10', 'Hit@10', 'Coverage@10']].round(4), hide_index=True)
    st.info('Honest reading: the collaborative models beat the item-mean baseline but are no better than a user-bias + movie-bias baseline, and the deployed blend only ties plain popularity on P@10. '
            'The diversify option raises catalogue coverage at a small accuracy cost.')

with tab_about:
    st.markdown("""
**What this is.** A demonstration of collaborative filtering (item-based, user-based and SVD) built for a learning project.

**Synthetic data.** Ratings beyond the 473 rows in the provided file were generated by a script with an AI assistant; they do not describe real people.
User IDs are pseudonymous codes. No personal or demographic data is used.

**Filter bubbles.** Recommenders tend to repeat what is already popular. In the audit the deployed model showed only a small share of the catalogue
to everyone; the *Diversify* option is a mitigation, and the diversity numbers on this page show its effect.

**Limitations.** Sparse data (about 13 ratings per user), personalisation adds little beyond user and movie averages, small test set. See docs/evaluation_report.md.
""")
