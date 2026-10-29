# Demo script (target length: 1 min 45 s)

Setup before recording: `streamlit run app/dashboard.py` in one terminal, `python api/app.py` in another. Browser zoom 110%. Start with user 8, method *Item CF + popularity (deployed)*, Diversify *Off*.

| Time | Show | Say |
|---|---|---|
| 0:00-0:12 | Title slide or dashboard top (with the synthetic-data banner) | "This is a movie recommender using collaborative filtering. Viewers get lost in big catalogues, and popularity lists hide regional films and create filter bubbles. All the data here is synthetic." |
| 0:12-0:32 | Recommendations tab, user 8 | "For an existing user the model predicts a rating for every unseen movie and shows the top ten, with genre, language and predicted rating. The charts compare this list's genre and language mix with the catalogue." |
| 0:32-0:55 | Switch Diversify to *Safe* | "Now I turn on the diversity re-ranker. The list becomes more varied: the diversity score rises. Across all users it doubled catalogue coverage from 15.9% to 31.0% for a 2.7% relative drop in precision at 10." |
| 0:55-1:10 | *Why this movie?* tab | "Each prediction is explained: global mean plus user bias plus movie bias plus a small adjustment from similar movies. Most of the accuracy comes from the biases." |
| 1:10-1:22 | Viewer = *New user (cold start)* | "A new user has no history, so the system falls back to the most-rated movies, re-ranked for variety." |
| 1:22-1:35 | Terminal: `curl "http://127.0.0.1:5000/api/recommend?user_id=8&k=3"` then `...user_id=abc` | "The same engine is a REST API, with friendly errors for bad input. Median response time is 3.0 milliseconds." |
| 1:35-1:50 | *How good is it?* tab | "Honest results on the sealed test set: collaborative filtering beats the item-mean baseline, but not a simple bias baseline, and the deployed blend only ties plain popularity. The value is a reproducible, tested pipeline with a working diversity control." |

Tips: keep it under 2 minutes; do not scroll fast; if a command fails on camera, re-record rather than explain.
