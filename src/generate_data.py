"""Stage 3 - Build a consistent movie catalogue and extend dataset.csv to 5,000 ratings.

SYNTHETIC DATA. Modelled on the structure of MovieLens 100K (users x movies x ratings, long-tail
popularity, user/item rating biases); no real MovieLens rows are copied.

Run from the repo root:  python src/generate_data.py
Outputs (data/raw/):
  movie_catalogue.csv          one fixed genre/language/year per title
  dataset_extended.csv         first 7 columns identical to the provided dataset.csv, plus movie_id, is_original
  synthetic_taste_groups.csv   hidden generator ground truth, ONLY for the Stage-10 fairness audit, never a model input
"""
import io
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
RAW = Path('data/raw')
TARGET_ROWS = 5000
MIN_PER_USER, MAX_PER_USER = 15, 25
START, END = pd.Timestamp('2024-01-01'), pd.Timestamp('2025-12-31 23:59:59')
GENRES = ['Action', 'Comedy', 'Drama', 'Romance', 'Sci-Fi', 'Thriller']

# title|year|primary genre  (ONE primary genre per film: a simplification, documented in docs/data_source.md)
CATALOGUE_TXT = {
'Hindi': '''3 Idiots|2009|Comedy
Andhadhun|2018|Thriller
Dangal|2016|Drama
Pink|2016|Drama
Drishyam|2015|Thriller
Lagaan|2001|Drama
Dilwale Dulhania Le Jayenge|1995|Romance
Queen|2013|Comedy
PK|2014|Comedy
Zindagi Na Milegi Dobara|2011|Drama
Gully Boy|2019|Drama
Barfi!|2012|Romance
Dil Chahta Hai|2001|Comedy
Rang De Basanti|2006|Drama
Taare Zameen Par|2007|Drama
Munna Bhai M.B.B.S.|2003|Comedy
Chak De! India|2007|Drama
Kahaani|2012|Thriller
Badla|2019|Thriller
Special 26|2013|Thriller
A Wednesday|2008|Thriller
Jab We Met|2007|Romance
Yeh Jawaani Hai Deewani|2013|Romance
Rockstar|2011|Romance
Veer-Zaara|2004|Romance
Koi... Mil Gaya|2003|Sci-Fi
Krrish|2006|Sci-Fi
Ra.One|2011|Sci-Fi
Bajrangi Bhaijaan|2015|Drama
Sholay|1975|Action
Dhoom 2|2006|Action
War|2019|Action
Pathaan|2023|Action
Jawan|2023|Action
Uri: The Surgical Strike|2019|Action
Bhool Bhulaiyaa|2007|Comedy
Hera Pheri|2000|Comedy
Stree|2018|Comedy
Tumbbad|2018|Thriller
Article 15|2019|Drama''',
'Tamil': '''Vikram|2022|Action
Master|2021|Action
Kaithi|2019|Action
Enthiran|2010|Sci-Fi
2.0|2018|Sci-Fi
24|2016|Sci-Fi
Vikram Vedha|2017|Thriller
Super Deluxe|2019|Drama
96|2018|Romance
Jai Bhim|2021|Drama
Asuran|2019|Drama
Soorarai Pottru|2020|Drama
Ratsasan|2018|Thriller
Pariyerum Perumal|2018|Drama
Mersal|2017|Action
Theri|2016|Action
Minnale|2001|Romance
Vinnaithaandi Varuvaayaa|2010|Romance
Maanagaram|2017|Thriller
Thani Oruvan|2015|Thriller
Love Today|2022|Comedy
Kolamaavu Kokila|2018|Comedy
Anniyan|2005|Thriller''',
'Telugu': '''Baahubali|2015|Action
RRR|2022|Action
Pushpa: The Rise|2021|Action
Magadheera|2009|Action
Athadu|2005|Action
Arjun Reddy|2017|Romance
Sita Ramam|2022|Romance
Geetha Govindam|2018|Romance
Dear Comrade|2019|Romance
Jersey|2019|Drama
Mahanati|2018|Drama
Rangasthalam|2018|Drama
Ala Vaikunthapurramuloo|2020|Comedy
Pelli Choopulu|2016|Comedy
F2|2019|Comedy
Brochevarevarura|2019|Comedy
Goodachari|2018|Thriller
Evaru|2019|Thriller
Kshanam|2016|Thriller
Agent Sai Srinivasa Athreya|2019|Thriller
Aditya 369|1991|Sci-Fi
Oke Oka Jeevitham|2022|Sci-Fi
Kalki 2898 AD|2024|Sci-Fi''',
'Kannada': '''KGF|2018|Action
Kantara|2022|Action
Ugramm|2014|Action
Kirik Party|2016|Comedy
Lucia|2013|Thriller
U Turn|2016|Thriller
Rangitaranga|2015|Thriller
Kavaludaari|2019|Thriller
Ulidavaru Kandanthe|2014|Drama
777 Charlie|2022|Drama
Mungaru Male|2006|Romance
Sapta Sagaradaache Ello|2023|Romance''',
'Malayalam': '''Premam|2015|Romance
Charlie|2015|Romance
Kumbalangi Nights|2019|Drama
Maheshinte Prathikaaram|2016|Comedy
Joji|2021|Thriller
Manjummel Boys|2024|Thriller
Minnal Murali|2021|Action
Lucifer|2019|Action
Android Kunjappan Version 5.25|2019|Sci-Fi''',
'English': '''Inception|2010|Sci-Fi
Interstellar|2014|Sci-Fi
The Matrix|1999|Sci-Fi
Blade Runner 2049|2017|Sci-Fi
The Martian|2015|Sci-Fi
The Dark Knight|2008|Action
Mad Max: Fury Road|2015|Action
Gladiator|2000|Action
Titanic|1997|Romance
La La Land|2016|Romance
The Notebook|2004|Romance
The Shawshank Redemption|1994|Drama
Forrest Gump|1994|Drama
Se7en|1995|Thriller
Shutter Island|2010|Thriller
Gone Girl|2014|Thriller
Knives Out|2019|Thriller
The Hangover|2009|Comedy
Superbad|2007|Comedy''',
}

# Latent taste groups (NOT demographics). Affinities in [-1, 1] for genre and language.
LANGS = list(CATALOGUE_TXT)
TASTE = {
 'Bollywood mainstream': ({'Comedy': .6, 'Romance': .6, 'Drama': .4},                {'Hindi': 1, 'English': -.3, 'Tamil': -.2, 'Telugu': -.2}),
 'Southern action':      ({'Action': 1, 'Thriller': .4},                              {'Telugu': 1, 'Tamil': .9, 'Kannada': .5}),
 'Thriller and mystery': ({'Thriller': 1, 'Drama': .3},                               {'Hindi': .3, 'Malayalam': .5, 'Kannada': .4, 'English': .3}),
 'Hollywood sci-fi':     ({'Sci-Fi': 1, 'Action': .4},                                {'English': 1, 'Hindi': -.2}),
 'Regional drama':       ({'Drama': 1, 'Romance': .6},                                {'Kannada': .8, 'Malayalam': .8, 'Tamil': .5}),
 'Broad explorer':       ({g: .3 for g in GENRES},                                    {l: .3 for l in LANGS}),
}
TASTE_P = [.25, .20, .15, .12, .15, .13]


def build_catalogue():
    rows = []
    for lang, txt in CATALOGUE_TXT.items():
        for line in txt.strip().splitlines():
            t, y, g = line.split('|')
            rows.append((t, g, lang, int(y)))
    cat = pd.DataFrame(rows, columns=['movie_title', 'genre', 'language', 'release_year'])
    assert cat.movie_title.is_unique and set(cat.genre) <= set(GENRES)
    cat.insert(0, 'movie_id', ['M%03d' % (i + 1) for i in range(len(cat))])
    return cat


def rand_time(rng, day):
    return pd.Timestamp(day) + pd.Timedelta(seconds=int(rng.integers(0, 86400)))


def main():
    rng = np.random.default_rng(SEED)
    cat = build_catalogue()
    cat.to_csv(RAW / 'movie_catalogue.csv', index=False)
    n_items = len(cat)

    # 1) original rows: drop duplicate (user, title) pairs keeping the latest, re-map attributes, add a time of day
    orig = pd.read_csv(RAW / 'dataset.csv')
    n_raw = len(orig)
    orig = orig.sort_values('timestamp').drop_duplicates(['user_id', 'movie_title'], keep='last')
    n_dup = n_raw - len(orig)
    orig = orig[['user_id', 'movie_title', 'rating', 'timestamp']].merge(cat, on='movie_title', how='left', validate='m:1')
    assert orig.movie_id.notna().all(), 'original title missing from catalogue'
    orig['timestamp'] = [rand_time(rng, d) for d in orig.timestamp]
    orig['is_original'] = True

    users = np.sort(orig.user_id.unique())
    n_users = len(users)
    uidx = {u: i for i, u in enumerate(users)}
    midx = {m: i for i, m in enumerate(cat.movie_id)}

    # 2) latent structure
    names = list(TASTE)
    G = np.array([[TASTE[n][0].get(g, 0) for g in GENRES] for n in names])
    L = np.array([[TASTE[n][1].get(l, 0) for l in LANGS] for n in names])
    mix = rng.dirichlet(np.array(TASTE_P) * 2.4, size=n_users)        # sparse mixtures of taste groups
    u_gen, u_lang = mix @ G, mix @ L
    g_of = cat.genre.map({g: i for i, g in enumerate(GENRES)}).to_numpy()
    l_of = cat.language.map({l: i for i, l in enumerate(LANGS)}).to_numpy()
    aff = u_gen[:, g_of] + u_lang[:, l_of]                              # users x items
    popularity = (np.arange(1, n_items + 1) ** -0.7)[rng.permutation(n_items)]   # Zipf-like long tail
    b_u = rng.normal(0, .35, n_users)
    b_i = rng.normal(0, .40, n_items) + .12 * (np.log(popularity) - np.log(popularity).mean()) / np.log(popularity).std()
    P, Q = rng.normal(size=(n_users, 4)), rng.normal(size=(n_items, 4))
    true = 3.45 + b_u[:, None] + b_i[None, :] + .8 * aff + .3 * (P @ Q.T) / 2.0

    # 3) how many ratings per user (15..25); originals count toward it; total forced to TARGET_ROWS
    n_u = np.clip(MIN_PER_USER + np.floor(rng.exponential(1.4, n_users)).astype(int), MIN_PER_USER, MAX_PER_USER)
    n_orig = orig.groupby('user_id').size().reindex(users).fillna(0).astype(int).to_numpy()
    diff = TARGET_ROWS - n_u.sum()
    while diff != 0:
        i = int(rng.integers(n_users))
        if diff > 0 and n_u[i] < MAX_PER_USER:
            n_u[i] += 1; diff -= 1
        elif diff < 0 and n_u[i] > max(MIN_PER_USER, n_orig[i]):
            n_u[i] -= 1; diff += 1
    need = n_u - n_orig

    # 4) sample unseen movies (popularity x taste fit), rate, timestamp
    seen = {(r.user_id, r.movie_id) for r in orig.itertuples()}
    min_date = np.where(cat.release_year.to_numpy() >= 2024, pd.Timestamp('2024-09-01').value, START.value)
    rows = []
    for u, i in uidx.items():
        w = popularity * np.exp(1.0 * aff[i])
        for m in [m for (uu, m) in seen if uu == u]:
            w[midx[m]] = 0
        w = w / w.sum()
        picks = rng.choice(n_items, size=need[i], replace=False, p=w)
        win_start = START + pd.Timedelta(days=int(rng.integers(0, 600)))
        for j in picks:
            r = float(np.clip(np.round((true[i, j] + rng.normal(0, .6)) * 2) / 2, 1, 5))
            t = win_start + pd.Timedelta(days=float(rng.uniform(0, 300)))
            if t > END or t.value < min_date[j]:
                t = pd.Timestamp(int(rng.integers(min_date[j], END.value)))
            rows.append((u, cat.movie_id[j], r, t.floor('s')))
    gen = pd.DataFrame(rows, columns=['user_id', 'movie_id', 'rating', 'timestamp']).merge(cat, on='movie_id')
    gen['is_original'] = False

    # 5) assemble: first 7 columns identical (names + order) to provided dataset.csv
    cols = ['user_id', 'movie_title', 'genre', 'language', 'release_year', 'rating', 'timestamp', 'movie_id', 'is_original']
    out = pd.concat([orig[cols], gen[cols]], ignore_index=True).sort_values('timestamp', kind='stable').reset_index(drop=True)
    out['timestamp'] = out.timestamp.dt.strftime('%Y-%m-%d %H:%M:%S')
    out.to_csv(RAW / 'dataset_extended.csv', index=False)

    groups = pd.DataFrame({'user_id': users, 'primary_taste_group': [names[k] for k in mix.argmax(1)]})
    groups.to_csv(RAW / 'synthetic_taste_groups.csv', index=False)
    print(f'raw rows {n_raw} | duplicate pairs dropped {n_dup} | originals kept {len(orig)} | generated {len(gen)}')
    print(f'final rows {len(out)} | users {out.user_id.nunique()} | movies in catalogue {n_items}, rated {out.movie_id.nunique()}')


if __name__ == '__main__':
    main()
