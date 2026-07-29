# Improving the model — a working guide

Where we are: **LightGBM, holdout Macro-F1 0.7514.** The whole six-model zoo
spans 0.6773–0.7529, and the top two are statistically tied.

This is a guide to *process*, not a list of tricks. The competition warns
"DO NOT OVER-ENGINEER YOUR SOLUTION", so the hard part is not finding ideas —
it is telling a real gain from a lucky one.

---

## Step 0 — The rule that governs everything

**A change is only an improvement if it beats the noise floor.**

On the 3000-row holdout, the bootstrap standard error of Macro-F1 is
**0.0084**. So:

| Observed gain | What it means |
|---|---|
| < 0.008 | Noise. Do not keep it on this evidence alone. |
| 0.008 – 0.017 | Suggestive. Re-test with more folds/seeds before believing it. |
| > 0.017 (2 SE) | Real. Keep it. |

For reference, the entire gap between LightGBM and XGBoost is **0.0015** —
one-fifth of a standard error. That is why they are treated as tied.

Always compare on the **same folds** (paired), and use McNemar's test for the
final call. `task3/best_model.py` already does both.

**Keep a log.** One row per experiment:

| # | Change | Holdout | Δ vs base | Public LB | Kept? |
|---|---|---|---|---|---|
| 0 | LightGBM baseline | 0.7514 | — | | — |

The `Public LB` column is the one that matters. If holdout keeps rising while
LB stalls, stop — you are fitting the holdout.

---

## Step 1 — Free wins, no new ideas required (do these first)

These cost compute, not cleverness, and cannot overfit.

### 1a. More seeds in the average
Currently 3. Push to 10.

```bash
.venv/bin/python task3/best_model.py --models lightgbm --seeds 10 --trials 20
```

Seed averaging is pure variance reduction — no decision is taken from
validation data, so it **cannot** overfit. Expect +0.002–0.008. Cheap at
LightGBM's ~100 s/fit.

### 1b. A real tuning budget
The reported run used only **10 trials per model** — chosen to keep six models
inside 3.5 hours, not because 10 is enough. LightGBM alone tunes in ~5 min.

```bash
.venv/bin/python task3/best_model.py --models lightgbm --trials 80 --folds 5 --repeats 2
```

`--repeats 2` halves the CV noise floor (√2), so the search chases signal
rather than fold luck. Expect +0.003–0.010.

### 1c. Check the threshold under shift
The OOF sweep chose 0.540 for LightGBM. That was optimised on
*in-distribution* data. Under a declared shift, the test-set optimum may sit
elsewhere, and Macro-F1 is sensitive to it.

Submit two files — one at the calibrated threshold, one at 0.5 — and compare
on the public LB. This is the single cheapest piece of information you can buy
about the shift, and it costs one extra submission.

---

## Step 2 — The most promising idea: features that survive the shift

**Hypothesis.** TF-IDF is a *vocabulary* representation. The test set draws on
generators and domains the training set does not cover, so its vocabulary
distribution differs — which is precisely what TF-IDF is most sensitive to.
Stylometric features (sentence length, punctuation rates, type-token ratio,
readability) describe *how* text is written rather than *which words* appear,
so they should degrade more gracefully across a domain shift.

This is testable, and it is the one idea here with a mechanism behind it rather
than hope.

**You already have the code:** `src/features_stylometric.py` produces 39
length-normalised features, pure numpy, no neural anything — fully compliant
with the Task 3 rule. And `data/train.csv` / `data/test.csv` carry the raw text.

```python
import pandas as pd, numpy as np, sys
sys.path.insert(0, "src")
from features_stylometric import extract_stylometric_matrix

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
S_tr, names = extract_stylometric_matrix(tr["text"].astype(str).tolist())
S_te, _     = extract_stylometric_matrix(te["text"].astype(str).tolist())
np.savez("data/stylometric.npz", tr=S_tr, te=S_te, names=names)
```

Then test **three** arms against the same folds:

| Arm | Features | Tests |
|---|---|---|
| A | TF-IDF only (baseline) | 0.7514 |
| B | TF-IDF + 39 stylometric | does adding them help? |
| C | stylometric only | how much signal is in style alone? |

Arm C is the interesting one. If it scores respectably (say >0.65) on 39
features versus 5000, that is strong evidence the style signal is real and
compact — and compact, domain-stable features are exactly what survives a
distribution shift. It would also be a genuinely interesting report finding
regardless of whether it wins.

**Declare it.** The brief permits your own feature engineering for Task 3 but
requires you to describe it, and awards no extra marks for it. Do it for the
score and the insight, not for marks.

> Note: two edits in `src/features_stylometric.py` need attention first. The
> `_PRONOUNS_2_OTHERS` / `_PRONOUNS_2_POETIC` sets are defined but never
> referenced, so they currently do nothing. And `_VOWELS` was changed from
> `"aeiouy"` to `"aeiou"`, which undercounts syllables ("typically" 4→2,
> "happy" 2→1) and shifts every readability feature. Restore the `y` unless
> you meant it.

---

## Step 3 — Reduce the dimensionality

Task 2 found KNN did far better on 100 PCA components than on 2000. Trees are
much less distance-sensitive, so the effect should be weaker — but 5000 sparse
columns is a lot of opportunity for spurious splits, and fewer inputs means
fewer chances to fit training-domain vocabulary.

Worth one experiment: `TruncatedSVD` (better than PCA on sparse data — it does
not centre, so it preserves sparsity) to 300–500 components, then LightGBM.

```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=400, random_state=50007)
X_tr_svd = svd.fit_transform(X_tr)     # fit on TRAIN FOLD only
X_va_svd = svd.transform(X_va)
```

Fit inside the fold, never on the full data — otherwise the validation set's
covariance leaks into the projection.

Prior: mild loss on holdout, possible gain on the leaderboard. Precisely the
kind of trade a shifted test set rewards and in-distribution CV cannot see.

---

## Step 4 — Squeeze the models you have

### Widen the regularisation search
The current LightGBM optimum picked `colsample_bytree=0.78` from a
0.25–0.8 range — it pinned against the ceiling, which means the range was too
narrow in that direction. Similarly `subsample=0.997`. Try 0.4–1.0 for both.

### Try CatBoost
The one major boosting family not yet tested. Its ordered-boosting scheme
resists target leakage on smaller datasets and it often behaves differently
enough to add blend value.

```bash
.venv/bin/pip install catboost
```

### Revisit the blend with fewer, more decorrelated members
The blend was rejected (McNemar p=0.587) but its weights were informative:
LightGBM 0.379, RandomForest 0.229, LinearSVC 0.217 — while XGBoost got 0.071
because it duplicates LightGBM. A three-member blend of
**LightGBM + LinearSVC + ComplementNB** (maximally decorrelated: boosting +
linear + generative) may beat the six-member one, since fewer weights means
less to overfit.

---

## Step 5 — What NOT to do

- **Don't chase the public leaderboard.** It is ~7000 rows of one draw. Trust
  your holdout and CV; a change that helps only on the LB is noise you paid for.
- **Don't add TF-IDF-derived features.** More vocabulary features cut against
  the shift, not with it.
- **Don't go deeper.** `max_depth` 3–8 already covers it; deeper memorises the
  training domain.
- **Don't tune to close the holdout↔LB gap.** That gap is designed into the
  dataset. The organisers said so.
- **Don't use a neural anything as the classifier.** `src/features_surprisal.py`
  needs GPT-2 and is barred from Task 3 — it stays research code.

---

## Suggested order

1. Seeds → 10, trials → 80 on LightGBM alone. *(~1 h, low risk)*
2. Submit calibrated-threshold vs 0.5. *(free, buys information about the shift)*
3. Build the stylometric features, run arms A/B/C. *(~2 h, highest upside)*
4. TruncatedSVD to 400 components. *(~1 h)*
5. CatBoost, then a 3-member decorrelated blend. *(~2 h)*

Stop when two consecutive experiments fail to clear 1 SE. Realistically the
ceiling here is around **0.76–0.77 holdout**, and the leaderboard will sit
below it by design.

The Task 4 rubric rewards the roadmap over the score. A well-documented
negative result — "we tested stylometric features expecting robustness to the
domain shift; they did not beat TF-IDF, and here is the paired test showing
it" — scores better than an unexplained 0.005.
