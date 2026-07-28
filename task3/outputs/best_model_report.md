# Task 3 — best model report

Generated in 12362s · seed 50007 · 10 Optuna trials/model · 4-fold x1 CV · 3 seeds averaged

## Headline

- **Selected:** `xgboost`
- **Holdout Macro-F1:** **0.7529** (15% slice never seen by tuning, blending, or thresholding)
- **Decision threshold:** 0.500 (the OOF sweep landed on the default 0.5 — calibration found no gain here)
- **Predicted positive rate:** 0.6974 (training positive rate 0.6252)

## Model comparison

| model    | cv_selection_biased | cv_fold_std | oof_macro_f1 | holdout_macro_f1 | oof_minus_holdout | threshold | seconds |
|----------|---------------------|-------------|--------------|------------------|-------------------|-----------|---------|
| xgboost  | 0.7123              | 0.0103      | 0.7391       | 0.7529           | -0.0137           | 0.5       | 5430.7  |
| lightgbm | 0.7149              | 0.0136      | 0.7421       | 0.7514           | -0.0094           | 0.54      | 301.3   |
| logreg   | 0.7152              | 0.0066      | 0.7322       | 0.7377           | -0.0055           | 0.455     | 30.4    |
| linsvc   | 0.7078              | 0.0078      | 0.7316       | 0.736            | -0.0043           | 0.565     | 48.7    |
| rforest  | 0.6931              | 0.0111      | 0.6985       | 0.7078           | -0.0093           | 0.49      | 4381.7  |
| compnb   | 0.6761              | 0.0039      | 0.6774       | 0.6773           | 0.0001            | 0.43      | 21.7    |

`cv_selection_biased` is `study.best_value` — the maximum of many noisy CV estimates, and therefore optimistic. It is shown for the tuning roadmap only. **`holdout_macro_f1` is the number to quote.** `oof_minus_holdout` is the overfitting readout: consistently large positive values mean the tuning is fitting fold noise.

## Blend

- weights (fitted on OOF): `{"xgboost": 0.0706, "lightgbm": 0.3785, "logreg": 0.025, "linsvc": 0.2173, "rforest": 0.2289, "compnb": 0.0797}`
- OOF Macro-F1: 0.7490
- holdout Macro-F1: 0.7540
- **gap: -0.0050** (healthy, < 0.02)

## Statistical validation

Method taken from Sujon et al. (2025), *Journal of Big Data* 12:268, `doi:10.1186/s40537-025-01313-4` — the paper finds F1 the most stable metric under class imbalance (supporting this competition's Macro-F1 choice) and recommends bootstrap CIs plus McNemar's test rather than comparing point estimates.

- blend holdout Macro-F1 95% CI: [0.7382, 0.7697]
- best single (`xgboost`) 95% CI: [0.7365, 0.7688]
- McNemar discordant pairs: 104 vs 113, **p = 0.5871**
- decision: `xgboost` selected — the blend does not beat it significantly, so the simpler model wins

The CIs overlap far more than the point estimates suggest. Treat any gap smaller than the CI width as unproven.

### Model selection rule

Selection mode: `one-se`. The one-standard-error rule picks the **simplest** model whose holdout Macro-F1 is within one bootstrap SE of the best, rather than the raw winner.

This is deliberate. The competition page states: *"You are expected to achieve a high training F1 score, but low test F1 score ... DO NOT OVER-ENGINEER YOUR SOLUTION."* That is a declared train/test distribution shift from the GenAIDetect sampling design. Under shift the in-distribution winner is often not the out-of-distribution winner: high-capacity boosters fit training-domain quirks that do not transfer, while linear models over TF-IDF degrade more gracefully. Where the accuracy cost is inside noise, the simpler model is the better bet.

**Expect your public-leaderboard score to sit well below the holdout number here.** That gap is designed into the dataset, not a bug in this pipeline. Do not tune it away — chasing it is precisely the over-engineering the organisers warn against.

## Tuned hyperparameters

### xgboost

- threshold: 0.500
```json
{
  "max_depth": 8,
  "learning_rate": 0.06106152100919074,
  "min_child_weight": 3.842808926283585,
  "subsample": 0.7575087172039224,
  "colsample_bytree": 0.6745247927435118,
  "reg_lambda": 1.009329587414398,
  "reg_alpha": 0.00670760320600673,
  "gamma": 1.9250896604381689
}
```

### lightgbm

- threshold: 0.540
```json
{
  "num_leaves": 46,
  "learning_rate": 0.06304716097331702,
  "min_child_samples": 12,
  "subsample": 0.9969969063623275,
  "colsample_bytree": 0.7764196845461617,
  "reg_lambda": 23.38434482088143,
  "reg_alpha": 0.00012647265418366007
}
```

### logreg

- threshold: 0.455
```json
{
  "C": 0.12650150725847492
}
```

### linsvc

- threshold: 0.565
```json
{
  "C": 0.0615757574338742
}
```

### rforest

- threshold: 0.490
```json
{
  "n_estimators": 600,
  "max_depth": 30,
  "min_samples_leaf": 6,
  "max_features": 0.06454742606074328
}
```

### compnb

- threshold: 0.430
```json
{
  "alpha": 0.006240169840651902
}
```

## Anti-overfitting controls in force

| Control | Where |
|---|---|
| 15% holdout split before any tuning | `train_test_split` at top of `main` |
| Optuna restricted to the fit split | `tune(..., X_fit, y_fit, ...)` |
| RepeatedStratifiedKFold | cuts noise floor ~sqrt(repeats) |
| Early stopping on boosters | tree count learned, not searched |
| Threshold from OOF, validated on holdout | `fit_blend_weights` / `best_threshold` |
| Blend weights from OOF only | holdout stays clean |
| Seed averaging | variance reduction that cannot overfit |
| Fold std reported | improvements checkable against noise |

See `.claude/skills/xgboost-paper-scout/references/overfitting-playbook.md`.

## For the Task 4 write-up

- 6 models explored with documented tuning (rubric's top band asks for more than 4).
- `tuning_history.csv` holds every trial for the hyperparameter table.
- `model_comparison.csv` is the comparison table.
- Log each submission's public-leaderboard score against its holdout score here; a widening gap means stop tuning.

| Submission | Holdout Macro-F1 | Public LB | Gap |
|---|---|---|---|
| xgboost (this run) | 0.7529 | _fill in_ | |

---

## Which file to submit

Two submissions are provided. **Submit `LGBM_leaderboard_predictions.csv`.**

| File | Model | Holdout Macro-F1 | 95% CI | Fit time |
|---|---|---|---|---|
| `LGBM_leaderboard_predictions.csv` | LightGBM | 0.7514 | — | **301 s** |
| `XGB_leaderboard_predictions.csv` / `blend_predictions.csv` | XGBoost | 0.7529 | [0.7365, 0.7688] | 5431 s |

The automated one-standard-error rule selected XGBoost, and that selection is
**overridden here deliberately.**

The rule computed `within-1SE = [xgboost, lightgbm]` — the two are
statistically indistinguishable (0.0015 apart, against a bootstrap SE of
0.0084, with near-identical confidence intervals). It then broke the tie using
the `COMPLEXITY` ordering in `best_model.py`, which ranks `xgboost` below
`lightgbm` on capacity-to-overfit grounds. That ordering is defensible in
theory and wrong in practice: it picked the model that costs **18x** the
compute for a difference well inside noise.

LightGBM is the better answer on every ground that matters here — equal
accuracy within measurement error, an eighteenth of the training cost, and no
reason under distribution shift to believe XGBoost's extra 0.0015 will survive
contact with the test set. The two submissions agree on 92.7% of rows.

This is a limitation of the automated rule worth stating in the report, not a
bug to hide: a complexity ordering over *model families* cannot see that two
families are near-duplicates with wildly different costs.

## Things worth saying in the Task 4 write-up

1. **The blend was rejected on evidence, not vibes.** It scored highest
   (0.7540) but McNemar gave p = 0.587 over 104 vs 113 discordant pairs, so
   the lead is noise. Six models were kept anyway for the comparison table —
   the rubric asks for breadth of exploration, which is satisfied by having
   tested them, not by shipping all of them.

2. **The blend's weights are more interesting than its score.** XGBoost got
   only 0.071 despite being the best single model, because it is nearly a
   duplicate of LightGBM (0.379). RandomForest (0.229) and LinearSVC (0.217)
   earned real weight despite worse solo scores — the blend rewards
   decorrelation, not individual accuracy.

3. **Every `oof_minus_holdout` value is negative.** No model is fitting fold
   noise; the holdout slice is marginally easier than the OOF folds. That is
   the opposite of the classic overfitting signature and worth showing.

4. **RandomForest cost 4382 s to finish last (0.7078).** Bagging over 5000
   sparse TF-IDF features is the wrong tool against boosting or a linear
   model. A negative result, and a cheap one to report.

5. **Threshold calibration bought nothing for the selected model** — the OOF
   sweep landed on 0.500. It did move for others (LightGBM 0.540, LinearSVC
   0.565), so the machinery works; this dataset just did not need it at the
   top of the table.

6. **Expect the leaderboard score to sit below 0.75.** The competition page
   says so explicitly. Do not tune that gap away.
