#!/usr/bin/env python3
"""
DEMO TEMPLATE — <PAPER TITLE>
=============================

Paper   : <full title>
Authors : <authors>
Venue   : <venue, year>   |   DOI/arXiv: <id>
PDF     : papers/pdf/<file>.pdf

ONE-LINE CLAIM
--------------
<What the paper says its technique does, in one sentence, in your own words.>

WHAT IS BORROWED
----------------
<The single specific mechanism being tested. Not the whole paper.>

COURSE COMPLIANCE
-----------------
<`OK` / `PARTIAL` / adapted-how. The 50.007 brief forbids deep learning and
LLMs for Task 3 — state explicitly why this demo complies.>

---------------------------------------------------------------------------
The contract this demo honours
---------------------------------------------------------------------------
1. Baseline and treatment are compared on the SAME folds (paired), so the
   difference is not confounded by how the data happened to split.
2. The Kaggle test set is never touched. Nothing here reads test labels,
   because there are none.
3. The gain is reported against fold-to-fold noise. A mean improvement
   smaller than the paired standard error is NOT a result.
4. Every seed is fixed and printed.

Run:  python3 demo_<slug>.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_SPLITS = 5
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. Data — the course-provided features. Same source as task1/task2/task3.
# ---------------------------------------------------------------------------

def load_data():
    """Returns (X, y, feature_names). Uses the competition's preprocessed
    features; swap in raw text if the paper's technique needs it."""
    from pathlib import Path

    import kagglehub

    root = Path(kagglehub.competition_download("50-007-machine-learning-may-2026"))
    train = pd.read_csv(root / "train_features.csv")

    label_col = next(c for c in ("label", "Label_A", "target", "y")
                     if c in train.columns)
    id_cols = [c for c in train.columns if c.lower() in ("id", "index")]
    y = train[label_col].astype(int).to_numpy()
    X = train.drop(columns=[label_col] + id_cols)
    return X.to_numpy(dtype=np.float64), y, list(X.columns)


# ---------------------------------------------------------------------------
# 2. Baseline — what you already have. Be honest here; a weak baseline makes
#    any technique look good and teaches you nothing.
# ---------------------------------------------------------------------------

BASELINE_PARAMS = dict(
    objective="binary:logistic", eval_metric="logloss", tree_method="hist",
    max_depth=6, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
    reg_lambda=10.0, random_state=SEED, n_jobs=-1,
)


def fit_baseline(X_tr, y_tr, X_va, y_va):
    import xgboost as xgb
    m = xgb.XGBClassifier(**BASELINE_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m.predict_proba(X_va)[:, 1]


# ---------------------------------------------------------------------------
# 3. THE PAPER'S TECHNIQUE — the only part that should differ.
#
#    Keep every other knob identical to the baseline. If you change the
#    technique AND the hyperparameters, you cannot attribute the difference.
# ---------------------------------------------------------------------------

def fit_treatment(X_tr, y_tr, X_va, y_va):
    """<Describe the mechanism. Cite the paper's section/equation number.>"""
    import xgboost as xgb

    params = dict(BASELINE_PARAMS)
    # --- BEGIN paper-specific change ---------------------------------------
    # e.g. params.update(max_delta_step=1, scale_pos_weight=...)
    # e.g. custom objective, custom feature block, resampling, calibration...
    # --- END paper-specific change -----------------------------------------

    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m.predict_proba(X_va)[:, 1]


# ---------------------------------------------------------------------------
# 4. Threshold calibration — applied identically to both arms so it cannot
#    flatter one of them. See references/overfitting-playbook.md §3.
# ---------------------------------------------------------------------------

def best_threshold(y_true, proba):
    grid = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, (proba >= t).astype(int), average="macro")
              for t in grid]
    return float(grid[int(np.argmax(scores))]), float(np.max(scores))


# ---------------------------------------------------------------------------
# 5. Paired evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    X, y, names = load_data()
    print(f"[data] X={X.shape}  positives={y.mean():.3f}  seed={SEED}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    base_scores, treat_scores = [], []

    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        X_tr, X_va, y_tr, y_va = X[tr], X[va], y[tr], y[va]

        p_base = fit_baseline(X_tr, y_tr, X_va, y_va)
        p_treat = fit_treatment(X_tr, y_tr, X_va, y_va)

        # Calibrate within the fold — a threshold picked on the full data
        # would leak across folds.
        _, f_base = best_threshold(y_va, p_base)
        _, f_treat = best_threshold(y_va, p_treat)

        base_scores.append(f_base)
        treat_scores.append(f_treat)
        print(f"  fold {k}: baseline={f_base:.4f}  paper={f_treat:.4f}  "
              f"delta={f_treat - f_base:+.4f}")

    base = np.array(base_scores)
    treat = np.array(treat_scores)
    diff = treat - base

    # Paired standard error — the right noise floor, because the folds are
    # shared. Comparing two independent means would overstate the noise.
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")

    print("\n" + "=" * 66)
    print(f"baseline Macro-F1 : {base.mean():.4f} +/- {base.std(ddof=1):.4f}")
    print(f"paper    Macro-F1 : {treat.mean():.4f} +/- {treat.std(ddof=1):.4f}")
    print(f"mean delta        : {diff.mean():+.4f}  (paired SE {se:.4f})")
    print("=" * 66)

    if diff.mean() > 2 * se:
        verdict = "REAL — gain exceeds 2x the paired standard error"
    elif diff.mean() > se:
        verdict = "WEAK — within 1-2 SE; re-run with more folds/repeats"
    elif diff.mean() > 0:
        verdict = "NOISE — positive but inside the noise floor"
    else:
        verdict = "NO GAIN on this dataset"
    print(f"verdict: {verdict}")
    print("\nA negative result is a result. Report it — the Task 4 rubric "
          "rewards the roadmap, not only the wins.")


if __name__ == "__main__":
    main()
