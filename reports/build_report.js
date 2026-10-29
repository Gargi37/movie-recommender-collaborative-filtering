// Builds reports/Project_Report.docx from reports/report_data.json.  Run from the repo root: node reports/build_report.js
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
} = require('docx');

const D = JSON.parse(fs.readFileSync('reports/report_data.json', 'utf8'));
const f = D.facts, T = D.tables, FIG = D.figs;
const NAVY = '1F3A5F', GREY = '6B7280', CONTENT_W = 9026;
const FONT = 'Arial';

// ---------- helpers
function runs(text, base = {}) {
  return String(text).split('**').map((seg, i) => new TextRun({ text: seg, bold: i % 2 === 1 || base.bold, font: FONT, size: base.size || 21, color: base.color, italics: base.italics }));
}
const P = (text, o = {}) => new Paragraph({ children: runs(text, o), spacing: { after: o.after ?? 120, line: 276 }, alignment: o.align, keepNext: o.keepNext });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], keepNext: true });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })], keepNext: true });
const B = (t) => new Paragraph({ numbering: { reference: 'bullets', level: 0 }, children: runs(t), spacing: { after: 70, line: 270 } });
const caption = (t) => new Paragraph({ children: [new TextRun({ text: t, font: FONT, size: 18, italics: true, color: GREY })], spacing: { before: 40, after: 180 }, alignment: AlignmentType.CENTER });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });
const space = (n = 120) => new Paragraph({ children: [], spacing: { after: n } });

function image(name, widthPx = 590) {
  const g = FIG[name];
  const w = Math.min(widthPx, 590), h = Math.round(w * g.h / g.w);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 40 }, keepNext: true,
    children: [new ImageRun({ type: 'png', data: fs.readFileSync(g.path), transformation: { width: w, height: h }, altText: { title: name, description: name, name } })] });
}
const figure = (name, cap, w) => [image(name, w), caption(cap)];

const border = { style: BorderStyle.SINGLE, size: 4, color: 'D1D5DB' };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, w, o = {}) {
  return new TableCell({ borders, width: { size: w, type: WidthType.DXA }, margins: { top: 60, bottom: 60, left: 100, right: 100 },
    shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR, color: 'auto' } : undefined,
    children: [new Paragraph({ children: runs(text, { size: 18, bold: o.bold, color: o.color }), alignment: o.align })] });
}
function table(t, opts = {}) {
  const widths = t.widths, total = widths.reduce((a, b) => a + b, 0);
  const head = new TableRow({ tableHeader: true, children: t.head.map((h, i) => cell(h, widths[i], { fill: NAVY, bold: true, color: 'FFFFFF' })) });
  const rows = t.rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(String(c), widths[i], { fill: ri % 2 ? 'F3F4F6' : undefined, bold: opts.boldFirst && i === 0 })) }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...rows] });
}
const tbl = (key, cap, opts) => [table(T[key], opts), caption(cap)];
function kv(rows, widths = [2600, 6426]) {
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map(([k, v]) => new TableRow({ children: [cell(k, widths[0], { fill: 'E5E7EB', bold: true }), cell(v, widths[1])] })) });
}
function placeholder(text) {
  const b = { style: BorderStyle.DASHED, size: 8, color: '9CA3AF' };
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W], rows: [new TableRow({ children: [new TableCell({
    borders: { top: b, bottom: b, left: b, right: b }, width: { size: CONTENT_W, type: WidthType.DXA }, margins: { top: 300, bottom: 300, left: 200, right: 200 },
    shading: { fill: 'FEF3C7', type: ShadingType.CLEAR, color: 'auto' },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: text, font: FONT, size: 19, italics: true, color: '92400E' })] })] })] })] });
}
function code(text) {
  return text.split('\n').map((l) => new Paragraph({ children: [new TextRun({ text: l === '' ? ' ' : l, font: 'Courier New', size: 17 })], spacing: { after: 0, line: 240 },
    shading: { fill: 'F3F4F6', type: ShadingType.CLEAR, color: 'auto' }, indent: { left: 120 } }));
}

const ins = D.insights;
const body = [];
const add = (...x) => x.flat().forEach((e) => body.push(e));

// ---------- title page
add(space(1500),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: 'HCLTech Career Shaper Programme', font: FONT, size: 26, color: GREY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [new TextRun({ text: 'III Year Industry Project 2025-26  |  Project No. 28', font: FONT, size: 22, color: GREY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: 'Personalized Movie Recommendation System Using Collaborative Filtering', font: FONT, size: 44, bold: true, color: NAVY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 900 }, children: [new TextRun({ text: 'Individual Project Report', font: FONT, size: 28, color: GREY })] }),
  kv([['Learner name', '[FILL IN]'], ['Roll number', '[FILL IN]'], ['College / campus', '[FILL IN]'], ['Mentor name', '[FILL IN]'], ['Submission date', '[FILL IN]'], ['Domain', 'Digital-Data-AI / AIML']]),
  space(300), P('This report describes a demonstration system built on synthetic data. Results are not findings about real viewers. AI-tool assistance is disclosed in Appendix A.', { italics: true, color: GREY, align: AlignmentType.CENTER }),
  pageBreak());

// ---------- summary + contents
add(H1('Summary'),
  P(`Streaming viewers face catalogues too large to browse, and popularity lists bury regional-language films. This project builds a movie recommender with **user-based collaborative filtering, item-based collaborative filtering and SVD matrix factorisation**, evaluates it with RMSE and precision@K against baselines, audits it for filter bubbles and fairness, and serves it through a Flask REST API and a Streamlit dashboard.`),
  P(`The provided dataset (${f.raw_rows} ratings of ${f.raw_titles} movies) could not support collaborative filtering, so it was extended to **${f.rows} ratings from ${f.users} users on ${f.catalogue} films**, of which ${f.gen} rows are synthetic and ${f.orig} come from the provided file. All conclusions below therefore concern a synthetic dataset.`),
  P(`**Main results (test set, opened once).** Item-based CF reduced RMSE to ${f.rmse_item_cf} against ${f.rmse_item_mean} for the item-mean baseline (difference ${f.d_itemmean}), but it did not improve on a simple user-bias plus movie-bias baseline (${f.rmse_bias}; difference ${f.d_bias}). For top-10 ranking, a blend of CF and popularity reached P@10 ${f.p10_dep}, a statistical tie with plain popularity (${f.p10_pop}). The recommendations were highly concentrated: catalogue coverage was only ${f.cov_dep}. A diversity re-ranker chosen on validation data raised coverage to ${f.cov_safe} and list diversity from ${f.ild_dep} to ${f.ild_safe} at a ${f.loss_safe} relative loss in P@10.`),
  P(`**Honest conclusion.** On this sparse data (about ${f.pu_med} ratings per user), personalisation adds little beyond knowing which movies are generally liked and which users rate generously. The value of the project lies in a reproducible pipeline, a leakage-free evaluation, an uncertainty-aware comparison and a working diversity mitigation.`),
  H2('Contents'),
  ...['1  Introduction', '2  Data description', '3  Data preparation', '4  Exploratory data analysis', '5  Methodology and system design', '6  Implementation', '7  Results and validation', '8  Ethical considerations and diversity audit', '9  Conclusion and future work', '10  References', 'Appendix A  AI-tool usage disclosure', 'Appendix B  Reproducibility', 'Appendix C  Validation experiment table'].map((t) => P(t, { after: 40 })),
  pageBreak());

// ---------- 1 Introduction
add(H1('1  Introduction'),
  H2('1.1  Background and problem statement'),
  P('Streaming platforms offer thousands of titles. Viewers (**who**) struggle to find films they will enjoy (**what**) every time they browse (**when/where**), and simple popularity lists hide regional-language films and trap viewers in a filter bubble (**why it matters**). This project builds collaborative-filtering recommenders and audits them for relevance, variety and fairness (**how**).'),
  H2('1.2  Target user and objectives'),
  P('The target user is a streaming-platform viewer who wants suggestions that are both relevant and varied, including Indian regional-language films. The objectives, taken from the project brief, are to implement user-based and item-based collaborative filtering and SVD matrix factorisation, evaluate them with RMSE and precision@K, include diversity metrics, avoid demographic bias, and deliver a working prototype.'),
  H2('1.3  Modules and success criteria'),
  B('**Modules:** data pipeline; user-based CF; item-based CF; SVD; evaluation and diversity audit; Flask REST API; Streamlit dashboard.'),
  B('**Success criteria** were fixed in Stage 1 and are checked against the final test results in Section 7.3 (Table 14).'),
  B('**Scope:** in scope are the dataset extension, three CF models, baselines, evaluation, audit, API, dashboard and report. Out of scope are real user accounts, deep-learning recommenders and online A/B testing.'));

// ---------- 2 Data
add(H1('2  Data description'),
  H2('2.1  The provided dataset and its problems'),
  P(`The provided dataset.csv has ${f.raw_rows} rows and 7 columns (user_id, movie_title, genre, language, release_year, rating, timestamp). Profiling showed that it cannot support collaborative filtering: only **${f.raw_titles} distinct titles**, ${f.raw_users} users with about 1.6 ratings each, ${f.raw_dup} duplicate user-movie pairs, date-only timestamps, no movie identifier, and every title appeared with several different genres, languages and release years (the attributes were random per row).`),
  H2('2.2  Responsible extension with synthetic data'),
  P('The Dataset Generation Prompt allows the data to be extended to 5,000 rows. It names MovieLens 100K as a source, but real MovieLens data was not downloaded in this project; only its structure (users x movies x ratings, long-tail popularity, user and item rating biases) was imitated. A seeded script (src/generate_data.py) therefore created a catalogue of 126 real film titles with one fixed genre, language and release year each, and generated ratings from a documented model.'),
  ...tbl('origin', 'Table 1: Origin of the 5,000 rows'),
  P('**Assumptions of the generator** (docs/data_source.md): each user is a sparse mixture of six latent taste groups (genre and language affinities, not demographics) plus four random latent factors; the movies a user rates depend on popularity (Zipf-like) and taste fit; ratings combine a base level, user bias, movie bias, taste affinity and noise, rounded to half stars; timestamps lie in 2024-2025 and never precede a film\'s release. The genre labels are one primary genre per film, assigned by the author and open to challenge. The original rows carry no taste signal and act as noise (about ' + f.orig_pct + ' of the data).'),
  ...tbl('data_stats', 'Table 2: Dataset summary after extension', { boldFirst: true }),
  P('A validation script (src/validate_data.py) checks 15 properties independently of the generator: column order, row count, missing values, duplicates, rating scale, ratings per user, attribute consistency, timestamp format and order, no rating before release, preservation of the original rows, and absence of PII. All pass.'),
  H2('2.3  Data dictionary'),
  P('data_dictionary.csv describes all nine columns: user_id (pseudonymous integer key), movie_title, genre (six values), language (six values), release_year, rating (1.0-5.0 in half steps; the prediction target), timestamp (used only to order ratings), movie_id (added key, M001-M126) and is_original (provenance flag, excluded from every model input). The full dictionary with type, unit, allowed values and notes is delivered as a separate file.'));

// ---------- 3 Data preparation
add(H1('3  Data preparation'),
  H2('3.1  Cleaning'),
  P('A repeatable function (src/clean.py) standardises text, parses types, checks domain rules, attaches film attributes from the catalogue, and removes duplicates, logging row counts at every step. Running it twice produces a byte-identical file.'),
  ...tbl('cleaning', 'Table 3: Cleaning log (5,000 rows in, 5,000 rows out)'),
  P('**No rows were removed**, because the generator already enforced these rules; the cleaning step does not invent problems. Seven unit tests feed deliberately dirty rows (spacing, casing, duplicates, bad ratings, missing values, bad dates, anachronistic ratings, unknown movies) to show that each rule works. No imputation is used: a missing rating cannot be estimated without inventing a preference.'),
  H2('3.2  Train, validation and test split'),
  P('The data is split **per user in time**: each user\'s latest 20% of ratings form the test set, and the latest 20% of the remainder form the validation set. A random split would let a model train on a user\'s later ratings and then "predict" earlier ones, which does not happen in production. Models may use only user_id, movie_id, rating and timestamp.'),
  ...tbl('splits', 'Table 4: Data splits'),
  B('**Leakage checks (asserted in code):** no user-movie pair appears in two sets; for every user all training ratings are earlier than that user\'s test ratings; indices, means and similarities are fitted on training data only.'),
  B('**Locked test set:** its SHA-256 hash was recorded before any modelling; the final evaluation script verifies the hash and refuses to run twice.'),
  B('**Cold start:** no unseen users or movies occur in this split (every user has at least 15 ratings, every movie at least 10), but a policy exists for the API: unseen user gets the movie mean, unseen movie the user mean, and unknown users receive a popularity list re-ranked for variety.'),
  B('**Limitation:** because the split is per user, other users\' later ratings can sit in the training data. This is the standard leave-latest-out protocol, not a strict global-time replay.'));

// ---------- 4 EDA
add(H1('4  Exploratory data analysis'),
  P('The EDA notebook (03_eda.ipynb) contains seven charts and five hypothesis tests. Each chart has a computed insight and a decision it drives. The most important are shown here.'),
  H2('4.1  Ratings'),
  ...figure('eda_01_rating_distribution', 'Figure 1: Distribution of ratings', 380),
  P('**Insight:** ' + ins[0].insight), P('**Decision:** ' + ins[0].decision),
  H2('4.2  Popularity and activity'),
  ...figure('eda_02_long_tail', 'Figure 2: Ratings per movie (ranked) and per user', 560),
  P('**Insight:** ' + ins[1].insight), P('**Decision:** ' + ins[1].decision),
  H2('4.3  Sparsity and overlap'),
  ...figure('eda_03_matrix_sparsity', 'Figure 3: Sample of the user-item matrix (white = not rated)', 500),
  P('**Insight:** ' + ins[2].insight), P('**Decision:** ' + ins[2].decision),
  H2('4.4  Languages and genres'),
  ...figure('eda_04_genre_language_share', 'Figure 4: Catalogue share versus ratings share', 560),
  P('**Insight:** ' + ins[3].insight), P('**Decision:** ' + ins[3].decision),
  H2('4.5  Time and rating bias'),
  ...figure('eda_06_ratings_over_time', 'Figure 5: Monthly ratings and mean rating', 500),
  P('**Insight:** ' + ins[5].insight), P('**Decision:** ' + ins[5].decision),
  ...figure('eda_07_user_movie_bias', 'Figure 6: Average rating given by users and received by movies', 560),
  P('**Insight:** ' + ins[6].insight), P('**Decision:** ' + ins[6].decision),
  H2('4.6  Hypothesis tests'),
  ...tbl('hyp', 'Table 5: Hypotheses (significance level 0.05)'),
  ...D.hyp_results.map((t) => B(t)),
  P('**Honesty note:** the generator deliberately built in user bias, movie bias, taste and popularity. Confirming them shows that the data behaves as designed and has something for the models to learn; it is **not** evidence about real viewers.'));

// ---------- 5 Methodology
add(H1('5  Methodology and system design'),
  H2('5.1  Architecture'),
  ...figure('arch_dfd_level0', 'Figure 7: Level-0 data flow diagram', 520),
  ...figure('arch_block_diagram', 'Figure 8: System block diagram', 590),
  H2('5.2  Baselines'),
  P(`Five baselines set the minimum standard: global mean, user mean, item (movie) mean, a **bias baseline** (global mean plus damped movie bias plus damped user bias, damping ${f.lam_i} and ${f.lam_u}), and for ranking random, most-popular and shrunk top-rated lists. Cold-start policy applies to all of them.`),
  H2('5.3  Collaborative-filtering models'),
  B('**Item-based CF:** movie-movie similarity is the cosine of baseline residuals over co-rating users, multiplied by a shrinkage factor n/(n + shrink) so pairs with few common raters count less. A prediction is the baseline plus a similarity-weighted average of the user\'s residuals on the k most similar movies, with a damping term in the denominator.'),
  B('**User-based CF:** the same idea over users, with Pearson or cosine similarity on co-rated movies, shrinkage and damping.'),
  B('**SVD matrix factorisation:** biased matrix factorisation trained by stochastic gradient descent with the Surprise library (factors, learning rate, regularisation, epochs).'),
  P('**A design flaw found and fixed:** the first item-CF version divided by the sum of similarities, which cancels their size, so one weak neighbour moved a prediction as much as a strong one. It was no better than the item-mean baseline. A sanity check (huge damping must reproduce the baseline exactly) exposed the problem, and a damping term fixed it. Both versions are kept in the experiment log.'),
  H2('5.4  Tuning protocol'),
  P('Models were tuned on **two rolling, time-respecting folds drawn from training data only** (fold 1: the first 75% of each user\'s training ratings to the next 25%; fold 2: train_core to validation). The selection rule, declared before the grids ran, was the lowest mean RMSE over the two folds. The bias baseline was tuned first and its damping reused by the CF models so the comparison is fair. A NaN bug at zero shrinkage was found and fixed, and grids were widened once when winners sat on grid edges.'),
  ...tbl('tuning', 'Table 6: Tuning grids and winners'),
  P('All tuned models are within about 0.001 RMSE of each other and of the tuned bias baseline, far below the noise of about 900 scored ratings. Item-based CF\'s best settings drift towards heavy shrinkage and light damping, that is, towards the baseline. By the pre-declared rule, item-based CF is the nominal winner and is the deployed model; SVD is nominally better on validation alone, and all tuned models are compared again on the test set without reselecting.'),
  ...tbl('lc', 'Table 7: Learning curve (validation RMSE; negative difference = better than baseline)'),
  P('SVD beats the baseline at every size but only by 0.003-0.010, and the gap does not grow with more data, so the report does not claim that more data would help.'),
  H2('5.5  Ranking: popularity blend and diversity re-ranker'),
  P(`Pure CF ranked by predicted rating scored below plain popularity on validation. A **CF + popularity blend** (z-scored CF score plus alpha x z-scored log popularity) was therefore tried; alpha = ${f.alpha} was chosen on the two folds by P@10. Because the blend still recommended almost the same movies to everyone, a **greedy MMR re-ranker with an exposure penalty** was added: score = lambda x relevance - (1 - lambda) x similarity to already chosen movies - gamma x exposure, applied to each user\'s top-50 candidates. Two operating points were chosen on validation by the rule "maximise catalogue coverage subject to a relative P@10 loss of at most 5% (safe) or 20% (diverse)": safe = lambda ${f.lam_safe}, gamma ${f.gam_safe}; diverse = lambda ${f.lam_div}, gamma ${f.gam_div}.`),
  ...figure('s10_03_accuracy_diversity_tradeoff', 'Figure 9: Validation trade-off between accuracy and catalogue coverage for re-ranker settings', 470));

// ---------- 6 Implementation
add(H1('6  Implementation'),
  H2('6.1  Technology stack'),
  ...tbl('stack', 'Table 8: Technology stack', { boldFirst: true }),
  H2('6.2  Service layer and REST API'),
  P('A shared service layer (src/serving.py) loads the saved models and implements recommend, similar, history and explain. Both the Flask API and the Streamlit dashboard call it, so they cannot disagree. Inputs are validated strictly (whole numbers only, k from 1 to 50, allowed method and mode values) and every error is a friendly JSON message with the right HTTP status; no traceback reaches the client.'),
  ...tbl('api', 'Table 9: REST endpoints'),
  P('Example request and response (real output):', { keepNext: true }), ...code('GET /api/recommend?user_id=8&k=3&diversify=safe\n' + D.api_sample), space(80),
  P('Example error response for invalid input:', { keepNext: true }), ...code('GET /api/recommend?user_id=abc\n' + D.api_error), space(80),
  P('**Cold start:** an unknown user still receives HTTP 200 with the most-rated movies, re-ranked for variety, and a note explaining the fallback. **Explanations:** /api/explain breaks a predicted rating into global mean, user bias, movie bias and a similar-movie adjustment, and lists the rated movies that contributed most.'),
  H2('6.3  Dashboard'),
  P('The Streamlit dashboard offers a user selector (or a new-user mode), a method selector, a diversify option, a list-diversity score, genre and language mix charts, a "Why this movie?" tab, the user\'s history, similar movies, the final test results and an ethics and limitations tab. A synthetic-data warning is always shown.'),
  placeholder('[INSERT SCREENSHOT 1: dashboard Recommendations tab for user 8 with Diversify = Safe]'), space(60),
  placeholder('[INSERT SCREENSHOT 2: "Why this movie?" tab showing the prediction breakdown]'), space(60),
  placeholder('[INSERT SCREENSHOT 3: browser or terminal showing /api/recommend response]'), space(120),
  H2('6.4  Testing and performance'),
  P(`${f.n_tests} automated tests pass. They cover the cleaning rules, the split (including leakage checks), similarity properties (Pearson checked against NumPy), the hand-computed ranking metrics, the bootstrap, the re-ranker, the explanation function, a guard that fails if any tuning script reads the test set, and the API: normal cases, boundaries (k = 1, 50, 0, 51), invalid input (letters, decimals, scientific notation, negatives, huge numbers, an SQL-injection string, a script tag), cold start, and a check that the API's safe mode reproduces the evaluated re-ranker exactly. The dashboard is run headless and every control is exercised.`),
  ...tbl('latency', 'Table 10: Measured API latency'),
  P(`Median response time for a top-10 recommendation was ${f.lat_med} ms (95th percentile ${f.lat_p95} ms) on a local machine; this measures the model and Flask only, not a network or production server.`),
  H2('6.5  How to run'),
  ...code('bash setup.sh\npython api/app.py                 # REST API on http://127.0.0.1:5000\nstreamlit run app/dashboard.py    # dashboard on http://localhost:8501\npython -m pytest -q tests         # all tests'));

// ---------- 7 Results
add(H1('7  Results and validation'),
  H2('7.1  Protocol'),
  P(`The final models were refit on train_full (${f.n_train_full} ratings) and scored once on the locked test set (${f.n_test} ratings; ${f.n_test_users} users with at least one relevant movie). Decisions made on validation data only: the deployed model, the blend weight and the re-ranker settings. Relevance for ranking means a rating of 4.0 or higher; unrated movies count as non-relevant, which is pessimistic. Uncertainty comes from a paired bootstrap over users (2,000 resamples).`),
  P(`${f.orig_test_share} of the test ratings come from the noisy original rows (RMSE ${f.orig_rmse} against ${f.gen_rmse} on generated rows), so test RMSE is slightly pessimistic; the tables also show RMSE without them.`),
  H2('7.2  Headline results'),
  ...tbl('test_rating', 'Table 11: Rating prediction on the test set'),
  ...tbl('test_rank', 'Table 12: Top-K ranking on the test set (Cov. = catalogue coverage over all users)'),
  ...figure('s10_01_rmse_differences', 'Figure 10: RMSE differences with 95% bootstrap intervals', 520),
  ...figure('s10_02_p10_differences', 'Figure 11: P@10 differences with 95% bootstrap intervals', 520),
  ...tbl('sig', 'Table 13: Paired bootstrap comparisons'),
  B(`Item-based CF clearly beats the item-mean and global-mean baselines, but is indistinguishable from the bias baseline (${f.d_bias}). SVD is nominally best (${f.rmse_svd}) but not significantly. Almost all accuracy comes from knowing which movies are liked and which users rate generously.`),
  B(`Pure CF ranking is significantly **worse** than popularity (${f.d_item_pop}). The blend ties popularity (${f.d_dep_pop}) and is significantly better than pure CF (${f.d_dep_item}).`),
  B('**Validation optimism:** on validation the blend had looked clearly better than popularity, yet on test the advantage disappeared. The blend weight was chosen on that same data, a concrete example of selection bias and the reason the test set was kept sealed. A plausible reason popularity is hard to beat: in this data users rate popular movies more often, so popularity predicts which movies get rated later, which is what the ranking test measures.'),
  H2('7.3  Success criteria'),
  ...tbl('crit', 'Table 14: Stage 1 criteria checked on the test set'),
  H2('7.4  Error analysis'),
  ...tbl('err', 'Table 15: RMSE of the deployed rating model by group'),
  ...figure('s10_06_error_analysis', 'Figure 12: Error analysis', 560),
  B(`Predictions are pulled towards the middle: movies actually rated 1-2 stars are predicted ${f.err_low} stars too high and movies rated 4.5-5 are predicted ${f.err_high} too low, typical of rating models on noisy data.`),
  B('Errors are similar across languages, genres, eras and activity levels; differences of a few hundredths are within noise at these group sizes.'),
  ...tbl('calib', 'Table 16: Calibration of predicted ratings'),
  H2('7.5  What drives the model'),
  B('Ablation: global mean, then adding the movie effect, then the user effect, then the CF neighbours (Table 11: RMSE falls from ' + f.rmse_global + ' to ' + f.rmse_item_mean + ', ' + f.rmse_bias + ' and ' + f.rmse_item_cf + '). Nearly all accuracy is in the biases.'),
  B(`SVD item factors explain ${f.svd_r2} of the variance by genre and language, against ${f.svd_null} for shuffled labels (chance level). The latent factors do **not** line up with genre or language and cannot honestly be called genre tastes.`),
  B(`Recommendations were also compared with the hidden generator taste groups (audit only, never a model input): alignment is ${f.align_dep} for the deployed model against ${f.align_rand} for random recommendations. **No model beats random**, so there is no evidence that the recommender recovers the built-in taste groups from this little data.`),
  B(`The deployed model gives a top-10 list with at least one later-liked movie for ${f.hit_dep} of users, against ${f.hit_rand} for random lists.`));

// ---------- 8 Ethics
add(H1('8  Ethical considerations and diversity audit'),
  H2('8.1  Filter bubbles and popularity bias'),
  P(`The programme brief warns that recommender systems can reinforce filter bubbles and asks for diversity metrics. The audit measures, for every user\'s top 10: intra-list diversity (share of movie pairs that differ in genre or language), catalogue coverage, the Gini index of exposure (0 = every movie shown equally), the share of recommendations from tail movies (the 101 least-rated of 126), novelty, unexpectedness and alignment with hidden taste.`),
  ...tbl('div', 'Table 17: Diversity and exposure audit (test set, top 10)'),
  ...figure('s10_05_exposure', 'Figure 13: Coverage, tail share and exposure Gini by model', 590),
  B(`The deployed model recommends only **${f.tail_dep} of 101 tail movies** (tail share ${f.tailshare_dep}); coverage is ${f.cov_dep} and the exposure Gini is ${f.gini_dep}. Almost everyone sees the same list. The popularity-heavy models score highest on P@10, while pure CF scores lower but is more varied, so accuracy and diversity pull in opposite directions.`),
  H2('8.2  Regional-language exposure'),
  P(`Tamil, Telugu, Kannada and Malayalam are ${f.reg_cat} of the catalogue and ${f.reg_rel} of relevant test ratings; the deployed model gives them ${f.reg_dep} of slots and the safe re-ranker ${f.reg_safe}. Within that, Telugu is over-exposed (${f.tel_dep} against ${f.tel_cat} of the catalogue) and English under-exposed (${f.eng_dep} against ${f.eng_cat}).`),
  ...tbl('lang', 'Table 18: Language shares of recommendations'),
  ...figure('s10_04_language_share', 'Figure 14: Language share of recommendations', 560),
  H2('8.3  Mitigation'),
  P(`The safe re-ranker, chosen on validation, raised catalogue coverage from ${f.cov_dep} to ${f.cov_safe} and combined list diversity from ${f.ild_dep} to ${f.ild_safe}, with P@10 falling from ${f.p10_dep} to ${f.p10_safe} (${f.loss_safe} relative; difference ${f.d_safe_dep}, interval includes zero). The diverse setting reaches coverage ${f.cov_div} at P@10 ${f.p10_div}. The exposure penalty depends on user processing order, so the API's "diverse" mode is an approximation; "safe" reproduces the evaluated procedure exactly. Coverage cannot exceed about 0.5 because only the top-50 candidates are re-ranked.`),
  H2('8.4  Fairness'),
  P('The dataset contains **no demographic attributes**, by design, so demographic bias cannot be measured. Fairness was assessed across user groups that do exist: activity level, rating generosity and the hidden generator taste group.'),
  ...tbl('fair', 'Table 19: Test RMSE by user group'),
  P(`The activity-level gap is ${f.gap_act}, below the 0.15 criterion. The taste-group gap (${f.gap_taste}) is about the same for the bias baseline (${f.gap_taste_bias}), so it reflects group noise and sample size (${f.taste_users} users per group), not something CF introduces. The bootstrap range of a max-min gap is biased upward.`),
  ...figure('s10_07_fairness_groups', 'Figure 15: Test RMSE by user group', 560),
  H2('8.5  Data ethics and privacy'),
  B('User identifiers are pseudonymous codes; the data has no names, contact details or demographics. Ratings beyond the 473 provided rows are synthetic and describe no real person, so no consent issue arises. The Digital Personal Data Protection Act 2023 applies to personal data and is not engaged by this dataset; a real deployment with real viewers would require consent, purpose limitation and data minimisation.'),
  B('The generated data and the use of AI assistance are disclosed (Section 2.2 and Appendix A). MovieLens is cited as the structural inspiration; none of its data is included, which respects its licence terms.'),
  B('Every API response and the dashboard state that the data is synthetic. Recommendations should not be presented as knowledge about real audiences.'));

// ---------- 9 Conclusion
add(H1('9  Conclusion and future work'),
  H2('9.1  Findings'),
  B(`A complete, reproducible pipeline was built: dataset extension, cleaning, leakage-free split, three CF models, baselines, tuning, a single sealed test evaluation, a diversity and fairness audit, a REST API and a dashboard, with ${f.n_tests} automated tests.`),
  B(`Collaborative filtering beats the item-mean baseline (${f.d_itemmean}) but not a user-bias plus movie-bias baseline on this sparse synthetic data.`),
  B(`The CF + popularity blend ties popularity on precision@10; the earlier validation advantage did not hold on test.`),
  B(`The accuracy-oriented recommender is very concentrated (coverage ${f.cov_dep}). The diversity re-ranker, chosen on validation, doubled coverage to ${f.cov_safe} for a ${f.loss_safe} relative loss in P@10.`),
  H2('9.2  Limitations'),
  B('Synthetic data: effects were built into the generator and results may not transfer to real viewers.'),
  B(`Sparse data (about ${f.pu_med} ratings per user): personalisation adds little beyond biases; the learning curve showed no growth with more data.`),
  B('The ranking test rewards predicting what users will rate, not only what they will like (popularity confound); unrated movies count as non-relevant.'),
  B(`About ${f.orig_test_share} of test ratings are noisy original rows; the test set is small (${f.n_test} ratings), so intervals are wide. The per-user split is not a strict global-time replay.`),
  B('Blend weight and re-ranker settings were chosen on validation, and the validation-to-test drop shows how optimistic such choices can be. There is no demographic data and no online A/B test.'),
  H2('9.3  Future work'),
  B('Repeat the study on real MovieLens data (100K or 1M), where density and signal are different, and add the regional-film ratings from a real source.'),
  B('Model implicit feedback (which movies get watched), add content features for cold-start movies, and try SVD++ or neural recommenders when more data is available.'),
  B('Evaluate exposure fairness for languages and creators with a proper exposure-parity metric, and test the diversity re-ranker with real users.'));

// ---------- 10 References
const refs = [
  'Harper, F. M. and Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19. https://grouplens.org/datasets/movielens/',
  'Koren, Y., Bell, R. and Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. Computer, 42(8), 30-37.',
  'Hug, N. (2020). Surprise: A Python library for recommender systems. Journal of Open Source Software, 5(52), 2174.',
  'Sarwar, B., Karypis, G., Konstan, J. and Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. Proceedings of the 10th International Conference on World Wide Web, 285-295.',
  'Carbonell, J. and Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. Proceedings of SIGIR 1998, 335-336.',
  'HCLTech Career Shaper Programme: Project Brief (Project 28), Dataset Generation Prompt and Programme Guidelines, provided by the Programme Coordinator. (Mentor to validate any additional links.)',
  'Documentation of the libraries used: pandas, NumPy, SciPy, Flask, Streamlit, matplotlib, seaborn and pytest.',
];
add(H1('10  References'), ...refs.map((r, i) => P(`[${i + 1}]  ${r}`, { after: 90 })));

// ---------- Appendices
add(pageBreak(), H1('Appendix A  AI-tool usage disclosure'),
  P('**[Learner: review this statement, make sure it is accurate for your own work, and follow the programme\'s rules on AI assistance.]**', { color: '92400E' }),
  P('An AI assistant (Claude, developed by Anthropic) was used throughout this project in a guided, stage-by-stage conversation following the programme\'s Industry Project Build Guide. The assistant wrote the Python code for the data-generation script, cleaning, splitting, models, tuning, evaluation, audit, API, dashboard and tests; produced the EDA notebook, the diagrams and the first drafts of the documentation and this report; ran the code and reported the outputs. All numbers in this report were produced by that code from the saved result files; none were typed by hand.'),
  P('The learner supplied the project brief, dataset and templates, directed the stage sequence, and is responsible for reviewing, running, understanding and being able to explain every part of the work, for checking the film genre labels in the generated catalogue, for capturing the screenshots, and for the declaration. The synthetic ratings were generated by a script written with the assistant; no rating rows were pasted from an AI chat. No real personal data was entered into any AI tool.'),
  P('Errors made by the assistant during the project (for example a faulty item-CF formula, a NaN bug at zero shrinkage, wrongly explained patterns that were then checked and corrected, and several report-generation bugs) were found through sanity checks and tests and are described in the relevant sections.'),
  H1('Appendix B  Reproducibility'),
  ...code('python src/generate_data.py          # catalogue + synthetic ratings (seed 42)\npython src/validate_data.py\npython src/clean.py\npython -m src.make_split             # splits + SHA-256 of the locked test set\npython -m src.tune                   # rolling-fold tuning (train data only)\npython -m src.diagnostics\npython -m src.finalize_stage9        # tuned models -> models/*.joblib\npython -m src.stage10 --split val    # debug + choose re-ranker on validation\npython -m src.stage10 --split test   # final evaluation: refuses to run twice\npython -m pytest -q tests'),
  space(100), P('Fixed seeds (42) are used everywhere. The experiment log is results/experiments.csv (rows E01-E29).'),
  H1('Appendix C  Validation experiment table'),
  ...tbl('val_exp', 'Table 20: Baselines and tuned models, fitted on train_core and scored on validation (Stages 8-9)'));

// ---------- document
const doc = new Document({
  creator: 'Learner', title: 'Personalized Movie Recommendation System Using Collaborative Filtering - Project Report',
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 32, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 25, bold: true, font: FONT, color: '374151' }, paragraph: { spacing: { before: 220, after: 110 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '\u2022', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, right: 1440, bottom: 1300, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: 'HCLTech Career Shaper Programme  |  Project 28  |  Individual project report', font: FONT, size: 16, color: GREY })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'Personalized Movie Recommendation System Using Collaborative Filtering  |  Page ', font: FONT, size: 16, color: GREY }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: GREY }), new TextRun({ text: ' of ', font: FONT, size: 16, color: GREY }), new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 16, color: GREY })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync('reports/Project_Report.docx', buf); console.log('report written', buf.length, 'bytes'); });
