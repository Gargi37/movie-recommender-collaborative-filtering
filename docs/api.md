# REST API reference (Stage 11)

Start: `python api/app.py` (default http://127.0.0.1:5000; set `PORT=5055` to change). Read-only: GET only. All data is **synthetic** (every response carries `data_note`).

| Endpoint | Parameters | Notes |
|---|---|---|
| `GET /api/health` | - | Status, counts, available methods and diversify modes |
| `GET /api/recommend` | `user_id` (required, 1..1e9), `k` (1-50, default 10), `method`, `diversify` | Top-k unseen movies |
| `GET /api/similar` | `movie` (movie_id such as `M001`, or exact title, case-insensitive), `n` (1-20, default 5) | Item-based CF neighbours |
| `GET /api/user/<id>/history` | `limit` (1-200, default 50) | Training ratings, newest first |
| `GET /api/explain` | `user_id`, `movie` | Prediction broken into global mean + user bias + movie bias + similar-movie adjustment |

**method**: `deployed` (default: item CF + popularity blend), `item_cf`, `user_cf`, `svd`, `svd_blend`, `popularity`.
**diversify**: `none`, `safe`, `diverse`. `safe` is exactly the MMR re-ranker evaluated in Stage 10 (lambda 0.4, gamma 0). `diverse` uses lambda 0.6, gamma 0.5 with a *static* exposure estimate, which approximates (does not exactly reproduce) the sequential batch procedure evaluated in Stage 10.

**Unknown users (cold start):** `/api/recommend` still answers (HTTP 200) with the most-rated movies, re-ranked for variety (`diversify` defaults to `safe`), `source: "popularity_fallback"` and an explanatory note. `method` is ignored.

## Examples
```bash
curl "http://127.0.0.1:5000/api/health"
curl "http://127.0.0.1:5000/api/recommend?user_id=8&k=5"
curl "http://127.0.0.1:5000/api/recommend?user_id=8&k=5&method=svd&diversify=safe"
curl "http://127.0.0.1:5000/api/recommend?user_id=99999"          # new user -> popularity fallback
curl "http://127.0.0.1:5000/api/similar?movie=Inception&n=3"
curl "http://127.0.0.1:5000/api/user/8/history?limit=5"
curl "http://127.0.0.1:5000/api/explain?user_id=8&movie=Pink"
```

## Errors
Always JSON: `{"error": {"code": "...", "message": "...", "hint": "..."}}`

| HTTP | code | When |
|---|---|---|
| 400 | `missing_parameter` | a required parameter is absent or empty |
| 400 | `invalid_parameter` | not a whole number, or not one of the allowed values |
| 400 | `out_of_range` | e.g. `k=0`, `k=51`, `user_id=0` |
| 404 | `user_not_found` / `movie_not_found` / `not_found` | no history or no match; unknown route |
| 405 | `method_not_allowed` | anything other than GET |
| 500 | `internal_error` | unexpected failure (details only in the server log) |

## Tests
`python -m pytest tests/test_api.py` covers normal, boundary and invalid cases, plus a check that `diversify=safe` reproduces the Stage 10 batch re-ranker exactly.
