"""Stage 8 demo of the item-based CF module on its own:  python -m src.demo_item_cf
Fits on train_full (the data a deployed system would have) and writes docs/demo_item_cf.md.
synthetic_taste_groups.csv is used ONLY to pick two contrasting demo users; it is never a model input."""
import pandas as pd
from src.item_cf import ItemCF

read = lambda n: pd.read_csv(f'data/processed/split/{n}.csv', parse_dates=['timestamp'])
train = read('train_full')
cat = pd.read_csv('data/raw/movie_catalogue.csv')
groups = pd.read_csv('data/raw/synthetic_taste_groups.csv')
model = ItemCF(k=20, shrink=10, damp=1.0, meta=cat).fit(train)
title = cat.set_index('movie_id').movie_title

L = ['# Demo: item-based collaborative filtering (Stage 8)', '',
     f'Model: ItemCF(k=20, shrink=10, damp=1.0), fitted on train_full ({len(train)} ratings, {model.nU} users, {model.nI} movies). Settings are untuned (Stage 9).', '']
fmt = lambda d: d.to_markdown(index=False) if hasattr(d, 'to_markdown') else d.to_string(index=False)

for grp in ['Hollywood sci-fi', 'Southern action']:
    u = int(groups[groups.primary_taste_group == grp].user_id.iloc[0])
    liked = train[(train.user_id == u) & (train.rating >= 4.5)].sort_values('rating', ascending=False).head(5)
    liked = liked.assign(movie=liked.movie_id.map(title))[['movie', 'genre', 'language', 'rating']]
    recs = model.recommend(u, k=5)[['movie_title', 'genre', 'language', 'score']]
    L += [f'## User {u} (generator taste group: {grp}, used only to choose a demo user)', '', '**Highest-rated movies in their history**', '', liked.to_string(index=False), '',
          '**Top-5 recommendations (unseen movies, by predicted rating)**', '', recs.to_string(index=False), '']

for m in ['Inception', 'Baahubali']:
    mid = cat.set_index('movie_title').movie_id[m]
    s = model.similar_items(mid, 5)[['movie_title', 'genre', 'language', 'similarity', 'co_raters']]
    L += [f'## Movies most similar to "{m}"', '', s.to_string(index=False), '']

L += ['## Reading the demo honestly',
      '- The module runs end to end: similarities, predictions, top-N lists that exclude seen movies.',
      '- Validation results (docs/experiments.md): with default settings it only marginally improves RMSE over the bias baseline and does not beat the popularity list on precision@10.',
      '- Item similarities rest on few co-raters (typically 2-3 per pair), so neighbours are noisy. Do not read the lists above as proof of taste understanding.']
text = '\n'.join('```' if False else l for l in L)
open('docs/demo_item_cf.md', 'w').write(text)
print(text)
