# XGBoost optimisation without overfitting

Ordered by how much score they buy per unit of risk, on *this* dataset:
~20k training rows, high-dimensional TF-IDF features, binary labels,
Macro-F1, and a public/private leaderboard split.

The governing fact: **the public leaderboard is ~2k–7k rows of a single
draw.** A 0.004 Macro-F1 gain that exists only there is noise you paid for
with generalisation. The brief says it outright — *"There is no perfection
100% Macro-F1 score (that's overfitting)."*

---

## 0. Three defects to fix before any tuning

Found in `task3_xgboost_leaderboard.py` as written. Each one silently
inflates the number you will report.

**a. Optuna selects on the same folds it reports.**
`run_optuna_search` returns `study.best_value` — the best of 50 CV scores —
and that number goes in the report. The maximum of 50 noisy estimates is a
biased estimate of the truth. With 5-fold CV on 20k rows the per-fold
standard error is roughly 0.005 Macro-F1, so the winner is typically
**0.005–0.015 optimistic** purely from selection.

Fix: hold out a slice Optuna never sees, and quote *that*.

```python
from sklearn.model_selection import train_test_split
X_fit, X_hold, y_fit, y_hold = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)
study = run_optuna_search(X_fit, y_fit, n_trials=50)   # never sees X_hold
final = xgb.XGBClassifier(**best_params).fit(X_fit, y_fit)
honest = f1_score(y_hold, final.predict(X_hold), average="macro")
# Report `honest`. Report study.best_value as "selection-biased CV" or not at all.
```

`train_hybrid.py` already does this correctly — it holds out 10% before
Optuna runs. Copy that structure.

**b. No early stopping.** `macro_f1_cv_score` fits the full `n_estimators`
every fold, so Optuna has to *guess* tree count as a hyperparameter. That
wastes trials and couples `n_estimators` to `learning_rate` in a way the
sampler cannot untangle. Let the data decide instead:

```python
model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_iter = model.best_iteration          # <- learned, not searched
```

Then drop `n_estimators` from the search space, fix it high (2000), and
tune `learning_rate` alone. You get a smaller space and a better fit.

**c. `output/` does not exist.** Lines 248/252/255 write into `output/`
with no `mkdir`. The run dies after the expensive search completes.
`Path("output").mkdir(parents=True, exist_ok=True)` at the top.

---

## 1. Trust CV over the leaderboard, and measure the gap

Keep a log of `(cv_score, public_lb_score)` for every submission. The gap
is your overfitting instrument:

| Pattern | Reading |
|---|---|
| CV ≈ LB (within ~0.01) | Healthy. Keep going. |
| CV ≫ LB, gap widening | You are fitting CV noise. Stop tuning, add regularisation. |
| CV < LB | Usually luck on an easy public split. Do not celebrate; the private split will correct it. |

When CV and LB disagree, **believe CV** — it averages 5 folds over 20k rows,
the public LB is one draw over far fewer.

Make CV worth trusting: `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)`
cuts the standard error by ~√3 for 3× the compute. On a dataset this size
that is minutes, and it is the single cheapest way to stop chasing noise.

---

## 2. The regularisation levers, strongest first

For wide sparse TF-IDF input specifically:

| Parameter | Range | What it does | Notes for this data |
|---|---|---|---|
| `min_child_weight` | 1–50, log | Min summed hessian per leaf | **Strongest lever here.** Default 1 lets a leaf form on a single rare token. Push to 5–30. |
| `colsample_bytree` | 0.3–0.8 | Features per tree | With thousands of TF-IDF columns, 0.3–0.5 often beats 0.8. Decorrelates trees. |
| `max_depth` | 3–8 | Interaction order | Text signal is mostly additive. Depth 4–6 usually wins; >8 memorises. |
| `reg_lambda` | 1–100, log | L2 on leaf weights | Safe to push hard. Start at 10, not the default 1. |
| `subsample` | 0.6–0.9 | Rows per tree | Cheap variance reduction. |
| `gamma` | 0–5 | Min loss drop to split | Blunt but effective when depth alone won't stop growth. |
| `reg_alpha` | 1e-8–10, log | L1 on leaf weights | Sparsifies; useful with many dead TF-IDF columns. |
| `learning_rate` | 0.02–0.1 | Step size | Lower + early stopping ≫ higher + fixed trees. |

The current script's `min_child_weight` range of 1–10 and
`colsample_bytree` floor of 0.5 are both too conservative for this feature
width. Widen them.

**`max_depth` and `min_child_weight` are the two that matter most.** Tune
them together first, fix them, then move to the rest — the staged approach
your report scaffold already describes.

---

## 3. Macro-F1 is not what XGBoost optimises

You train on `binary:logistic` (logloss) and are graded on Macro-F1. Those
disagree, and closing the gap is nearly free score.

**Threshold calibration.** The 0.5 cutoff is arbitrary. Sweep it on
held-out data:

```python
proba = model.predict_proba(X_hold)[:, 1]
ts = np.linspace(0.05, 0.95, 181)
best_t = max(ts, key=lambda t: f1_score(y_hold, (proba >= t).astype(int),
                                        average="macro"))
```

Typically worth **0.005–0.02 Macro-F1**. `train_hybrid.py` has this as
`calibrate_threshold()`; `task3_xgboost_leaderboard.py` does not — it calls
bare `.predict()`, hardcoding 0.5. Port it over.

Two rules: calibrate on data the model did not train on (otherwise the
threshold overfits too), and pick the threshold **inside each CV fold** if
you want an unbiased estimate of what calibration buys.

**`scale_pos_weight` interacts with the threshold.** Both shift the
decision boundary. Setting `scale_pos_weight = n_neg/n_pos` *and* then
calibrating can overshoot. Try `scale_pos_weight=1` with calibration and
compare — on mildly imbalanced data, calibration alone often wins.

---

## 4. Seed averaging — the safest gain available

Same params, several seeds, average the probabilities:

```python
probas = []
for seed in (0, 1, 2, 3, 4):
    m = xgb.XGBClassifier(**best_params, random_state=seed).fit(X, y)
    probas.append(m.predict_proba(X_te)[:, 1])
pred = (np.mean(probas, axis=0) >= best_t).astype(int)
```

This reduces variance without touching bias — it **cannot** overfit,
because no decision is being made from validation data. Reliably worth
0.002–0.008 Macro-F1 for 5× training time. Do this before any exotic
tuning.

---

## 5. Feature-count discipline

More features are not free. Every added column is another chance to find a
spurious split.

- Cap TF-IDF via `max_features` (5k–20k) and `min_df=2` — a term appearing
  once cannot generalise, it can only memorise.
- After a first fit, drop zero-importance features and refit. Usually
  neutral-to-positive on score and much faster.
- **Never** select features using the full labelled set and then
  cross-validate on it — that leaks the target into your feature set and
  inflates CV. Selection belongs inside the fold.

Same rule for anything fitted: PCA, scalers, TF-IDF vocabulary. Fit on the
training fold only. `sklearn.pipeline.Pipeline` inside CV enforces this
automatically; hand-rolled preprocessing usually gets it wrong.

> Note for Task 2: `task2_pca_knn.py` fits PCA on `X_tr` and transforms
> `X_val` — correct. Keep that discipline when reusing PCA features in
> Task 3.

---

## 6. Model portfolio (also worth marks)

Task 3's top rubric band needs **more than 4 models** with documented
tuning. Blending decorrelated models also genuinely beats any single one.

| Model | Why it adds something |
|---|---|
| XGBoost | Baseline; strongest single tabular learner |
| LightGBM | Leaf-wise growth — different bias, fast on wide data |
| CatBoost | Ordered boosting resists target leakage on small data |
| Linear SVM / LogReg | Linear on TF-IDF is a genuinely strong text baseline; decorrelates hard from trees |
| Random Forest | Bagged, not boosted — different error profile |
| Complement Naive Bayes | Near-free, built for imbalanced text |

Blend by **averaging probabilities** with weights fixed from CV, not from
the leaderboard. Stacking with a logistic meta-learner works too, but the
meta-learner must be trained on out-of-fold predictions or it leaks badly.

---

## 7. Diminishing returns — when to stop

- Optuna: gains flatten past ~50–100 trials on a space this size. Plot
  best-value-vs-trial; when the curve is flat for 20 trials, stop.
- If a change improves CV by less than one standard error across folds, it
  is not a real improvement. Print `np.std(fold_scores)` alongside the mean
  and hold yourself to it.
- The brief's own warning applies: this is <5% of the source dataset, so
  the achievable ceiling is well below the ~0.99 F1 the SKDU paper reports
  on full De-Factify data. Chasing that number *is* the overfitting failure
  mode.

---

## 8. Quick audit checklist

- [ ] Held-out slice that hyperparameter search never touched
- [ ] Reported score comes from that slice, not from `study.best_value`
- [ ] Early stopping on a validation fold; `n_estimators` not hand-searched
- [ ] All preprocessing fitted inside the CV fold
- [ ] Threshold calibrated on held-out data, not 0.5, not on train
- [ ] `RepeatedStratifiedKFold` (or fold-score std reported)
- [ ] Seed-averaged final prediction
- [ ] CV-vs-public-LB gap logged for every submission
- [ ] `output/` directory created before writing
- [ ] Every random_state fixed and stated in the report
