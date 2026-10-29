# Final submission checklist

| # | Deliverable (from the brief) | File / location | Status |
|---|---|---|---|
| 1 | Completed project (working prototype) | `movie-recsys-final.zip` (API, dashboard, models, tests) | Built and tested here. **You: run it on your own machine** (commands in README) |
| 2 | Project report, minimum 8 pages, individual | `Project_Report_YourName_RollNo.docx` (24 pages) | Drafted. **You: fill name/roll/college/mentor/date, add 3 screenshots, review every section, rename the file** |
| 3 | EDA notebook with 3+ visualisations | `03_eda.ipynb` (7 charts, 5 hypothesis tests) | Done |
| 4 | dataset.csv, cleaned and pre-processed | `dataset.csv` (5000 rows) | Done |
| 5 | data_dictionary.csv, all columns described | `data_dictionary.csv` (9 columns) | Done |
| 6 | Demo video (1-2 min) or live demo | script in `demo_script.md` | **You: record it** |
| 7 | Signed Ethical Compliance Declaration | section 9 of the synopsis + the programme form | **You: read, tick and sign. The assistant has not ticked or signed anything** |
| + | Project synopsis (template) | `Project_Synopsis_YourName_RollNo.docx` | Filled. **You: complete the placeholders; rename per the template** |

## Repository checks
- [x] README, requirements.txt, models, code and tests in the repository
- [x] 56 automated tests pass
- [x] No passwords, API keys or personal data in the repository (scanned)
- [x] Test set locked (hash in docs/split_report.md) and opened once (results/test_opened.txt)

## Your to-do list before submitting
1. Fill every `[FILL IN]` (report title page and synopsis table) and the submission date.
2. Spot-check the genre label of each film in `data/raw/movie_catalogue.csv` (assigned from memory; one primary genre per film).
3. Take the three screenshots (dashboard recommendations, "Why this movie?", API response) and paste them over the yellow placeholders in section 6.3 of the report.
4. Run `pip install -r requirements.txt` and the three run commands on your own machine (Windows users: activate the venv with `venv\Scripts\activate`). If scikit-surprise will not install, tell your mentor; a NumPy SVD fallback can be added.
5. Read Appendix A of the report (AI disclosure), make sure it is accurate, and check the programme's rules on AI assistance.
6. Ask your mentor to validate the references and approve the problem statement and EDA (brief checkpoints).
7. Make sure you can explain every part in a viva: the split, why the bias baseline matters, why popularity wins, what the re-ranker trades off, and why the data is synthetic.
8. Read, tick and sign the Ethical Compliance Declaration yourself.
