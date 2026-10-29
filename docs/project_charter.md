# Project Charter: Stage 1

**Title:** Personalized Movie Recommendation System Using Collaborative Filtering (No. 28)
**Target user:** A streaming-platform viewer who wants relevant AND varied movie suggestions, including Indian regional-language films.

## Problem statement (5W1H)
Viewers on streaming platforms (WHO) struggle to find films they will enjoy in a huge catalogue (WHAT), especially regional-language titles that popularity-based lists bury (WHEN/WHERE: every time they browse). This matters because poor recommendations reduce engagement and trap users in a filter bubble (WHY). I will build user-based, item-based and SVD collaborative-filtering recommenders, evaluated with RMSE and precision@K and audited for diversity and fairness, and serve them through a Flask API and a demo UI (HOW).

## Modules (from the brief description)
1. Data pipeline (generate, clean, validate)
2. User-based collaborative filtering
3. Item-based collaborative filtering
4. SVD matrix factorisation
5. Evaluation and diversity/fairness audit
6. Flask REST API
7. Demo UI

## Task types
Rating prediction (regression) + top-K ranking (recommendation).

## Success criteria (targets now; results filled in Stage 10)
| # | Criterion | Target | Result | Met? |
|---|-----------|--------|--------|------|
| 1 | Best CF RMSE vs item-mean baseline | Lower | TBD | |
| 2 | Best CF RMSE vs global-mean baseline | Lower | TBD | |
| 3 | Precision@10 vs popularity baseline | Higher | TBD | |
| 4 | Catalogue coverage @10 | Reported; target >= 30% | TBD | |
| 5 | Intra-list diversity (genre and language) | Reported; re-ranker improves it with <= 5% relative precision loss | TBD | |
| 6 | RMSE gap across user activity groups | <= 0.15 | TBD | |
| 7 | API and UI | All endpoint tests pass; documented run commands | TBD | |
Targets 4-6 are initial assumptions; adjust after baselines (Stage 8), before touching the test set.

## Scope
In: dataset extension, 3 CF models, baselines, evaluation, diversity audit, REST API, demo UI, report.
Out: real user accounts, deep-learning recommenders, online A/B testing, content-based features.

## Risks and ethical concerns
- Provided data: 8 titles, about 1.6 ratings per user, random genre/language/year per row, 27 duplicate (user, movie) pairs. It must be rebuilt as a consistent synthetic dataset (Stage 3).
- Synthetic data may not reflect real taste; state this honestly.
- Filter bubbles and popularity bias; mitigate with diversity re-ranking.
- No demographic data exists, so fairness is checked across activity and taste groups only.
- Disclose all AI-tool usage; cite MovieLens (Harper and Konstan, 2015).
