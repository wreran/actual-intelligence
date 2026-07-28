# actualintelligence

50.007 Machine Learning — Group Project. **GenAI content detection:** given a
piece of text, classify it as human-authored (`0`) or machine-generated (`1`).

Data is a <5% subsample of the COLING 2025 GenAIDetect Workshop set, provided
pre-processed: stop words removed, lemmatised, top **5000 TF-IDF features**.
20,000 train rows / 6,999 test rows, 62.5% positive. Scored on **Macro-F1**.

> **Read this before tuning anything.** The competition page states: *"You are
> expected to achieve a high training F1 score, but low test F1 score ... DO NOT
> OVER-ENGINEER YOUR SOLUTION."* The train/test gap is a deliberate consequence
> of how the organisers sampled the data — the test set draws on generators and
> domains the training set does not cover. A large CV-to-leaderboard gap is
> expected and is not a bug to be tuned away.

## Layout

```
data/                     competition CSVs (gitignored — see Setup)
docs/                     brief, research notes, verification report
papers/                   paper-scout output: shortlist, manifest, PDFs
src/                      shared feature-extraction library (research code)
task1/  logistic_regression.py   from-scratch logistic regression   [5 marks]
task2/  pca_knn.py               PCA + KNN sweep                    [5 marks]
task3/  best_model.py            six-model tuned pipeline          [15 marks]
        xgboost_leaderboard.py   earlier single-model attempt
task4/  report-scaffold.md       final report skeleton             [25 marks]
tests/                    smoke tests for the pure-math components
```

Every script resolves paths from its own location, so it runs the same from the
repo root or from inside its task directory. Outputs land in `task*/outputs/`.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Then put the five competition CSVs in `data/`:

```
data/train.csv  data/train_features.csv
data/test.csv   data/test_features.csv
data/sample_submission.csv
```

Either download them from the competition page by hand, or with credentials in
`~/.kaggle/kaggle.json`:

```bash
.venv/bin/python -c "import kagglehub,shutil,pathlib; \
  p=kagglehub.competition_download('50-007-machine-learning-may-2026'); \
  [shutil.copy(f,'data/') for f in pathlib.Path(p).glob('*.csv')]"
```

## Running

```bash
.venv/bin/python task1/logistic_regression.py     # -> task1/outputs/LogReg_predictions.csv
.venv/bin/python task2/pca_knn.py                 # -> task2/outputs/ (2000/1000/500/100 sweep)
.venv/bin/python task3/best_model.py --fast       # smoke test first
.venv/bin/python task3/best_model.py --trials 40  # full run
.venv/bin/python tests/test_smoke.py
```

`task3/best_model.py` trains XGBoost, LightGBM, LogisticRegression, LinearSVC,
RandomForest and ComplementNB, tunes each with Optuna, and blends them. It
writes `best_model_report.md` with every number the Task 4 write-up needs.

Design notes, and why it deliberately resists over-tuning, are in
`.claude/skills/xgboost-paper-scout/references/overfitting-playbook.md`.

## Task requirements at a glance

| Task | Requirement | Note |
|---|---|---|
| 1 | Logistic regression **from scratch** | No sklearn LR. Submission must be named `LogReg_predictions.csv`. |
| 2 | PCA + KNN, `n_neighbors=2` | Report Macro-F1 at 2000 / 1000 / 500 / 100 components. |
| 3 | Leaderboard | **No deep learning or LLMs.** Top rubric band needs >4 models with documented tuning. |
| 4 | Report + presentation | 15 + 10 marks. Learning process matters more than the score. |

Tasks 1 and 2 **must** use the provided `*_features.csv`. Task 3 may use your
own feature engineering, but it must be described in the report and earns no
extra marks.

## Research notes

`src/` and `docs/` hold an earlier research thread on hybrid stylometric +
surprisal detection. `docs/verification-report.md` fact-checks its claims
against primary sources and is worth reading before citing any of it — several
claims did not survive.

`src/features_surprisal.py` requires a frozen GPT-2. **That is not usable for
Task 3** under the no-deep-learning rule, and is kept only as research code.
