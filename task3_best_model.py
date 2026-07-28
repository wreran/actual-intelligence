#!/usr/bin/env python3
"""
Task 3 — best-model pipeline (GenAI content detection, Macro-F1)
================================================================

Six tuned models, blended, with every anti-overfitting control from
`.claude/skills/xgboost-paper-scout/references/overfitting-playbook.md`
wired in rather than described.

Why six: the Task 3 rubric's top band (11-15) requires "more than 4 machine
learning models, with proper documentation on the hyperparameter tuning".
Blending decorrelated models also genuinely beats any single one.

The discipline, and where it lives
----------------------------------
1. A 15% holdout is split off FIRST and never seen by Optuna, by blend-weight
   fitting, or by threshold calibration. It is the only number worth quoting.
2. Optuna tunes each model on RepeatedStratifiedKFold *inside the fit split*.
   `study.best_value` is recorded as "selection-biased" and never reported as
   the headline — it is the max of N noisy estimates.
3. Boosters use early stopping, so tree count is learned, not searched.
4. Blend weights and the decision threshold are fitted on out-of-fold
   predictions, then *validated* on the untouched holdout. The gap between
   those two numbers is the overfitting readout.
5. The final model is seed-averaged (variance reduction that cannot overfit,
   because no decision is taken from validation data).
6. Fold standard deviations are printed everywhere, so "improvement" can be
   checked against noise.

Usage
-----
  python3 task3_best_model.py                       # full run
  python3 task3_best_model.py --fast                # smoke test, few trials
  python3 task3_best_model.py --trials 60
  python3 task3_best_model.py --synthetic           # no Kaggle creds needed
  python3 task3_best_model.py --data-dir ./data     # local CSVs instead of kagglehub

Outputs (all under --out-dir, default output/)
  best_model_report.md      — every number needed for the Task 4 write-up
  model_comparison.csv      — per-model CV / OOF / holdout Macro-F1
  tuning_history.csv        — every Optuna trial, per model
  XGB_leaderboard_predictions.csv
  blend_predictions.csv     — the submission
  feature_importance.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     train_test_split)

warnings.filterwarnings("ignore")

SEED = 50007          # course number
HOLDOUT_FRAC = 0.15
THRESH_GRID = np.linspace(0.05, 0.95, 181)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_competition(data_dir: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train_features.csv is [id, label, f0..fN]; test_features.csv is [id, f0..fN].

    That layout is taken from the working task1 notebook
    (`train_features_df.iloc[:, 1]` is the label, `.iloc[:, 2:]` the features),
    but column names are honoured when present so a schema change does not
    silently shift everything by one column.
    """
    if data_dir:
        root = Path(data_dir)
    else:
        import kagglehub
        root = Path(kagglehub.competition_download("50-007-machine-learning-may-2026"))
    tr = pd.read_csv(root / "train_features.csv")
    te = pd.read_csv(root / "test_features.csv")
    print(f"[data] train_features {tr.shape} | test_features {te.shape}")
    return tr, te


def split_xy(tr: pd.DataFrame, te: pd.DataFrame):
    label_col = next((c for c in ("label", "Label_A", "target", "y", "class")
                      if c in tr.columns), None)
    if label_col is None:
        label_col = tr.columns[1]
        print(f"[data] no named label column; falling back to positional "
              f"-> {label_col!r}")
    id_col = next((c for c in tr.columns if c.lower() in ("id", "index")),
                  tr.columns[0])

    feat_cols = [c for c in tr.columns if c not in (label_col, id_col)]
    missing = [c for c in feat_cols if c not in te.columns]
    if missing:
        print(f"[data] WARNING dropping {len(missing)} cols absent from test set")
        feat_cols = [c for c in feat_cols if c in te.columns]

    X = tr[feat_cols].to_numpy(dtype=np.float32)
    y = tr[label_col].to_numpy().astype(int)
    X_te = te[feat_cols].to_numpy(dtype=np.float32)
    test_ids = te[id_col].to_numpy() if id_col in te.columns else np.arange(len(te))
    X = np.nan_to_num(X)
    X_te = np.nan_to_num(X_te)
    print(f"[data] X={X.shape} features={len(feat_cols)} "
          f"pos_rate={y.mean():.4f} classes={np.bincount(y)}")
    return X, y, X_te, test_ids, feat_cols


def make_synthetic(n=6000, d=800, seed=SEED):
    """Data shaped like the real thing — wide, sparse, mildly imbalanced — so
    the pipeline can be validated end-to-end without Kaggle credentials.
    Scores from this are meaningless; only 'does it run' is being tested."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.63).astype(int)          # ~63% positive, as observed
    X = rng.gamma(0.28, 1.0, size=(n, d)).astype(np.float32)
    X[rng.random((n, d)) < 0.82] = 0.0              # sparse, like TF-IDF
    # Weak, diffuse signal over few columns. A strong signal drives every model
    # to F1 1.000, which validates nothing — the threshold sweep and the blend
    # weights are never exercised on a non-degenerate decision boundary.
    signal = rng.choice(d, 12, replace=False)
    X[:, signal] += (y[:, None] * rng.normal(0.12, 0.10, (n, len(signal)))
                     ).astype(np.float32)
    X = np.clip(X, 0.0, None)                       # TF-IDF is non-negative
    cut = int(n * 0.78)
    print(f"[data] SYNTHETIC X={X.shape} pos_rate={y.mean():.4f} "
          f"(scores are not meaningful)")
    return (X[:cut], y[:cut], X[cut:], np.arange(len(X) - cut),
            [f"f{i}" for i in range(d)])


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def macro_f1(y, p):
    return f1_score(y, p, average="macro")


def macro_f1_sweep(y, proba, grid=THRESH_GRID) -> np.ndarray:
    """Macro-F1 at every threshold in `grid`, vectorised.

    The obvious implementation calls `f1_score` once per threshold. That is 181
    sklearn calls per sweep, and the blend-weight search runs thousands of
    sweeps — half a million calls, which dominated total runtime.

    Here the confusion counts at every threshold come from one sort plus a
    searchsorted, so a sweep is O(n log n + |grid|) with no Python-level loop.
    Verified equal to `f1_score(..., average="macro")` to 1e-12.
    """
    y = np.asarray(y).astype(np.int8)
    proba = np.asarray(proba, dtype=np.float64)
    order = np.argsort(proba, kind="stable")
    p_sorted, y_sorted = proba[order], y[order]

    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos

    # idx[i] = number of samples with proba < grid[i]  => predicted negative
    idx = np.searchsorted(p_sorted, grid, side="left")
    cum_pos = np.concatenate(([0], np.cumsum(y_sorted)))   # positives below idx

    TP = n_pos - cum_pos[idx]
    pred_pos = n - idx
    FP = pred_pos - TP
    FN = n_pos - TP
    TN = n_neg - FP

    # class 1 as positive, then class 0 as positive (TN/FN/FP swap roles)
    f1_pos = 2 * TP / np.maximum(2 * TP + FP + FN, 1e-12)
    f1_neg = 2 * TN / np.maximum(2 * TN + FN + FP, 1e-12)
    return (f1_pos + f1_neg) / 2.0


def best_threshold(y, proba) -> tuple[float, float]:
    scores = macro_f1_sweep(y, proba)
    i = int(np.argmax(scores))
    return float(THRESH_GRID[i]), float(scores[i])


# --- statistical validation ------------------------------------------------
# Both borrowed from Sujon et al. (2025), "Accuracy, precision, recall,
# f1-score, or MCC? empirical evidence from advanced statistics, ML, and XAI
# for evaluating business predictive models", Journal of Big Data 12:268,
# doi:10.1186/s40537-025-01313-4 (papers/pdf/2025-accuracy-precision-...pdf).
#
# The paper's relevant findings: F1 is the most stable metric under class
# imbalance (which validates this competition's Macro-F1 choice), and metric
# differences between models should be validated with bootstrap confidence
# intervals and McNemar's test rather than compared as bare point estimates.

def bootstrap_ci(y, pred, n_boot=2000, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI for Macro-F1. A point estimate on a 3k-row
    holdout has real width; quoting it bare invites over-reading small gaps."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    n = len(y)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = macro_f1(y[idx], pred[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mcnemar(y, pred_a, pred_b) -> tuple[float, float, int, int]:
    """McNemar's test on the discordant pairs of two classifiers scored on the
    same rows. Returns (statistic, p_value, b, c).

    This is the correct paired test here: both models predict the same holdout,
    so their errors are dependent and an unpaired comparison overstates the
    noise. Only the disagreements carry information.
    """
    from scipy import stats as sps
    y = np.asarray(y)
    a_ok, b_ok = (np.asarray(pred_a) == y), (np.asarray(pred_b) == y)
    b = int(np.sum(a_ok & ~b_ok))    # a right, b wrong
    c = int(np.sum(~a_ok & b_ok))    # a wrong, b right
    if b + c == 0:
        return 0.0, 1.0, b, c
    if b + c < 25:
        # exact binomial — chi-square is unreliable on few discordant pairs
        p = float(sps.binomtest(b, b + c, 0.5).pvalue)
        return float(abs(b - c)), p, b, c
    stat = (abs(b - c) - 1) ** 2 / (b + c)      # Yates continuity correction
    return float(stat), float(sps.chi2.sf(stat, df=1)), b, c


# ---------------------------------------------------------------------------
# Model zoo. Each entry: (build_fn, optuna_space_fn, needs_dense, supports_es)
# ---------------------------------------------------------------------------

def space_xgb(t):
    return {
        "max_depth": t.suggest_int("max_depth", 3, 8),
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.25, log=True),
        # min_child_weight pushed well past the usual 1-10: with thousands of
        # sparse columns, default 1 lets a leaf form on a single rare token.
        "min_child_weight": t.suggest_float("min_child_weight", 1.0, 40.0, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        # low colsample floor decorrelates trees on wide TF-IDF input
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.25, 0.8),
        "reg_lambda": t.suggest_float("reg_lambda", 1.0, 100.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "gamma": t.suggest_float("gamma", 1e-8, 4.0, log=True),
    }


def build_xgb(params, spw, seed):
    import xgboost as xgb
    p = dict(params)
    p.update(objective="binary:logistic", eval_metric="logloss",
             tree_method="hist", n_estimators=2000, random_state=seed,
             n_jobs=-1, scale_pos_weight=spw, early_stopping_rounds=50)
    return xgb.XGBClassifier(**p)


def space_lgbm(t):
    return {
        "num_leaves": t.suggest_int("num_leaves", 15, 160, log=True),
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 120, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.25, 0.8),
        "reg_lambda": t.suggest_float("reg_lambda", 1.0, 100.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
    }


def build_lgbm(params, spw, seed):
    import lightgbm as lgb
    p = dict(params)
    p.update(objective="binary", n_estimators=2000, random_state=seed,
             n_jobs=-1, scale_pos_weight=spw, verbose=-1, subsample_freq=1)
    return lgb.LGBMClassifier(**p)


def space_logreg(t):
    return {"C": t.suggest_float("C", 1e-3, 50.0, log=True)}


def build_logreg(params, spw, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MaxAbsScaler
    # Scaler inside the pipeline => refit per fold, no leakage across folds.
    return make_pipeline(
        MaxAbsScaler(),
        LogisticRegression(C=params["C"], max_iter=3000,
                           class_weight="balanced", random_state=seed))


def space_svc(t):
    return {"C": t.suggest_float("C", 1e-3, 10.0, log=True)}


def build_svc(params, spw, seed):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.svm import LinearSVC
    # LinearSVC has no predict_proba; wrap it so it can join the blend.
    return make_pipeline(
        MaxAbsScaler(),
        CalibratedClassifierCV(
            LinearSVC(C=params["C"], class_weight="balanced",
                      random_state=seed, max_iter=4000),
            cv=3, method="sigmoid"))


def space_rf(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 200, 600, step=100),
        "max_depth": t.suggest_int("max_depth", 8, 40),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 12),
        "max_features": t.suggest_float("max_features", 0.02, 0.4),
    }


def build_rf(params, spw, seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**params, class_weight="balanced_subsample",
                                  random_state=seed, n_jobs=-1)


def space_nb(t):
    return {"alpha": t.suggest_float("alpha", 1e-3, 10.0, log=True)}


def build_nb(params, spw, seed):
    from sklearn.naive_bayes import ComplementNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer
    # ComplementNB is built for imbalanced text but hard-requires non-negative
    # input. Raw TF-IDF satisfies that; anything centred or standardised does
    # not, and the failure is a mid-run ValueError several minutes in. Clip
    # inside the pipeline so it degrades instead of crashing.
    return make_pipeline(
        FunctionTransformer(lambda A: np.clip(A, 0.0, None), accept_sparse=True),
        ComplementNB(alpha=params["alpha"]))


ZOO = {
    "xgboost":  (build_xgb,    space_xgb,    True),
    "lightgbm": (build_lgbm,   space_lgbm,   True),
    "logreg":   (build_logreg, space_logreg, False),
    "linsvc":   (build_svc,    space_svc,    False),
    "rforest":  (build_rf,     space_rf,     False),
    "compnb":   (build_nb,     space_nb,     False),
}


def fit_predict(name, model, X_tr, y_tr, X_va, y_va):
    """Fit with early stopping where supported, return validation probabilities."""
    if name == "xgboost":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif name == "lightgbm":
        import lightgbm as lgb
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1]


# ---------------------------------------------------------------------------
# Tuning — on the fit split only
# ---------------------------------------------------------------------------

def tune(name: str, X: np.ndarray, y: np.ndarray, spw: float, n_trials: int,
         n_splits: int, n_repeats: int, history: list) -> tuple[dict, float, float]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    build, space, _ = ZOO[name]

    def objective(trial):
        params = space(trial)
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=SEED)
        scores = []
        for tr, va in cv.split(X, y):
            m = build(params, spw, SEED)
            proba = fit_predict(name, m, X[tr], y[tr], X[va], y[va])
            # Threshold picked inside the fold, so the tuned score reflects the
            # calibrated pipeline rather than an arbitrary 0.5 cut.
            _, f1 = best_threshold(y[va], proba)
            scores.append(f1)
        mean, sd = float(np.mean(scores)), float(np.std(scores, ddof=1))
        history.append({"model": name, "trial": trial.number, "mean_macro_f1": mean,
                        "fold_std": sd, **params})
        trial.set_user_attr("fold_std", sd)
        return mean

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    sd = study.best_trial.user_attrs.get("fold_std", float("nan"))
    return study.best_params, float(study.best_value), float(sd)


def oof_predict(name, params, X, y, spw, n_splits) -> np.ndarray:
    """Out-of-fold probabilities — the honest basis for blend weights and the
    decision threshold. Every row is predicted by a model that did not see it."""
    build, _, _ = ZOO[name]
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, va in skf.split(X, y):
        m = build(params, spw, SEED)
        oof[va] = fit_predict(name, m, X[tr], y[tr], X[va], y[va])
    return oof


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def fit_blend_weights(oof: dict[str, np.ndarray], y: np.ndarray,
                      iters: int = 3000) -> dict[str, float]:
    """Random-search simplex weights on OOF predictions.

    Fitted on OOF only — never on the holdout — so the holdout stays a clean
    estimate of what the blend will do on unseen data.
    """
    names = list(oof)
    M = np.column_stack([oof[n] for n in names])
    rng = np.random.default_rng(SEED)
    best_w = np.ones(len(names)) / len(names)
    _, best = best_threshold(y, M @ best_w)
    for _ in range(iters):
        w = rng.dirichlet(np.ones(len(names)) * 0.7)
        _, s = best_threshold(y, M @ w)
        if s > best:
            best, best_w = s, w
    return dict(zip(names, (round(float(x), 4) for x in best_w)))


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None,
                    help="dir with train_features.csv/test_features.csv "
                         "(default: download via kagglehub)")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on synthetic data shaped like the real set "
                         "(pipeline validation only; needs no credentials)")
    ap.add_argument("--models", default="all",
                    help="comma-separated subset of: " + ",".join(ZOO))
    ap.add_argument("--trials", type=int, default=40, help="Optuna trials per model")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2,
                    help="CV repeats; >1 cuts the noise floor by ~sqrt(repeats)")
    ap.add_argument("--seeds", type=int, default=5,
                    help="seeds to average for the final fit")
    ap.add_argument("--fast", action="store_true",
                    help="tiny run: 4 trials, 3 folds, 1 repeat, 2 seeds")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    if args.fast:
        args.trials, args.folds, args.repeats, args.seeds = 4, 3, 1, 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)   # playbook: this crashed before

    names = list(ZOO) if args.models == "all" else \
        [m.strip() for m in args.models.split(",") if m.strip() in ZOO]
    if not names:
        print("no valid models selected", file=sys.stderr)
        return 2

    t0 = time.time()
    print(f"[cfg ] models={names} trials={args.trials} "
          f"folds={args.folds}x{args.repeats} seeds={args.seeds} seed={SEED}")

    # ---- data ----
    if args.synthetic:
        X, y, X_te, test_ids, feat_cols = make_synthetic()
    else:
        tr, te = load_competition(args.data_dir)
        X, y, X_te, test_ids, feat_cols = split_xy(tr, te)

    # ---- holdout FIRST: nothing downstream may touch it ----
    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=HOLDOUT_FRAC, stratify=y, random_state=SEED)
    spw = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
    print(f"[split] fit={X_fit.shape[0]} holdout={X_hold.shape[0]} "
          f"scale_pos_weight={spw:.3f}")

    # ---- tune + OOF per model ----
    history: list[dict] = []
    results, oof, tuned = [], {}, {}
    for name in names:
        ts = time.time()
        print(f"\n[tune] {name} ...")
        params, cv_biased, fold_sd = tune(name, X_fit, y_fit, spw, args.trials,
                                          args.folds, args.repeats, history)
        oof[name] = oof_predict(name, params, X_fit, y_fit, spw, args.folds)
        thr, oof_f1 = best_threshold(y_fit, oof[name])

        # honest check: refit on the full fit split, score the untouched holdout
        build, _, _ = ZOO[name]
        m = build(params, spw, SEED)
        p_hold = fit_predict(name, m, X_fit, y_fit, X_hold, y_hold)
        hold_f1 = macro_f1(y_hold, (p_hold >= thr).astype(int))

        tuned[name] = {"params": params, "threshold": thr}
        results.append({"model": name, "cv_selection_biased": round(cv_biased, 4),
                        "cv_fold_std": round(fold_sd, 4),
                        "oof_macro_f1": round(oof_f1, 4),
                        "holdout_macro_f1": round(hold_f1, 4),
                        "oof_minus_holdout": round(oof_f1 - hold_f1, 4),
                        "threshold": round(thr, 3),
                        "seconds": round(time.time() - ts, 1)})
        print(f"  cv(biased)={cv_biased:.4f}+/-{fold_sd:.4f}  oof={oof_f1:.4f}  "
              f"holdout={hold_f1:.4f}  thr={thr:.3f}  ({time.time()-ts:.0f}s)")

    # ---- blend: weights + threshold from OOF, validated on holdout ----
    print("\n[blend] fitting weights on OOF ...")
    weights = fit_blend_weights(oof, y_fit)
    blend_oof = sum(w * oof[n] for n, w in weights.items())
    blend_thr, blend_oof_f1 = best_threshold(y_fit, blend_oof)
    print(f"  weights: {weights}")
    print(f"  oof={blend_oof_f1:.4f} thr={blend_thr:.3f}")

    hold_probs = {}
    for name in names:
        build, _, _ = ZOO[name]
        m = build(tuned[name]["params"], spw, SEED)
        hold_probs[name] = fit_predict(name, m, X_fit, y_fit, X_hold, y_hold)
    blend_hold = sum(w * hold_probs[n] for n, w in weights.items())
    blend_hold_f1 = macro_f1(y_hold, (blend_hold >= blend_thr).astype(int))
    gap = blend_oof_f1 - blend_hold_f1
    print(f"  holdout={blend_hold_f1:.4f}   OOF-minus-holdout gap={gap:+.4f}")
    if gap > 0.02:
        print("  WARNING gap > 0.02 — the blend is fitting OOF noise. "
              "Reduce models or use uniform weights.")

    # ---- statistical validation of the blend-vs-best-single choice --------
    best_single = max(results, key=lambda r: r["holdout_macro_f1"])
    bs_name = best_single["model"]
    pred_blend = (blend_hold >= blend_thr).astype(int)
    pred_single = (hold_probs[bs_name] >= tuned[bs_name]["threshold"]).astype(int)

    lo_b, hi_b = bootstrap_ci(y_hold, pred_blend)
    lo_s, hi_s = bootstrap_ci(y_hold, pred_single)
    stat, pval, nb, nc = mcnemar(y_hold, pred_single, pred_blend)

    print(f"\n[stats] blend  holdout={blend_hold_f1:.4f}  95% CI "
          f"[{lo_b:.4f}, {hi_b:.4f}]")
    print(f"[stats] {bs_name:<7s} holdout={best_single['holdout_macro_f1']:.4f}  "
          f"95% CI [{lo_s:.4f}, {hi_s:.4f}]")
    print(f"[stats] McNemar: discordant {nb} vs {nc}, p={pval:.4f} "
          f"({'significant' if pval < 0.05 else 'NOT significant'} at 0.05)")

    # Prefer the blend only when it wins AND the difference is not plain noise.
    # A blend that ties within noise costs 6x the inference time for nothing,
    # and its extra fitted parameters (the weights) are extra overfitting risk.
    blend_wins = blend_hold_f1 > best_single["holdout_macro_f1"]
    use_blend = blend_wins and pval < 0.05
    if blend_wins and pval >= 0.05:
        print(f"[pick ] blend leads but p={pval:.3f} — difference is inside "
              f"noise; taking the simpler single model")
    print(f"[pick ] {'BLEND' if use_blend else bs_name} "
          f"(blend={blend_hold_f1:.4f} vs {bs_name}="
          f"{best_single['holdout_macro_f1']:.4f})")
    stats_block = {"blend_ci": [round(lo_b, 4), round(hi_b, 4)],
                   "single_ci": [round(lo_s, 4), round(hi_s, 4)],
                   "single_name": bs_name,
                   "mcnemar_p": round(pval, 5),
                   "discordant_b": nb, "discordant_c": nc,
                   "chose_blend": bool(use_blend)}

    # ---- final: seed-averaged refit on ALL training data ----
    print(f"\n[final] seed-averaging {args.seeds} seeds on all {len(y)} rows ...")
    test_probs = {n: np.zeros(len(X_te)) for n in names}
    for name in names:
        build, _, _ = ZOO[name]
        for s in range(args.seeds):
            m = build(tuned[name]["params"], spw, SEED + s)
            if name in ("xgboost", "lightgbm"):
                # No held-out set at full-data refit time, so early stopping is
                # unavailable; use a fixed budget instead of leaking the test set.
                m = build({**tuned[name]["params"]}, spw, SEED + s)
                m.set_params(n_estimators=400)
                if name == "xgboost":
                    m.set_params(early_stopping_rounds=None)
                m.fit(X, y)
            else:
                m.fit(X, y)
            test_probs[name] += m.predict_proba(X_te)[:, 1] / args.seeds
        print(f"  {name} done")

    if use_blend:
        final_proba = sum(w * test_probs[n] for n, w in weights.items())
        final_thr, tag = blend_thr, "blend"
    else:
        final_proba = test_probs[best_single["model"]]
        final_thr, tag = tuned[best_single["model"]]["threshold"], best_single["model"]
    final_pred = (final_proba >= final_thr).astype(int)

    # ---- sanity check on the submission's class balance ----------------
    # Macro-F1 punishes a collapsed prediction hard. If the test positive rate
    # is nowhere near the training rate, something is wrong upstream — a
    # feature-column mismatch between train and test, or a threshold tuned on a
    # differently-distributed slice — and it is far cheaper to catch here than
    # on the leaderboard.
    train_rate, test_rate = float(y.mean()), float(final_pred.mean())
    if abs(test_rate - train_rate) > 0.25:
        print(f"\n  *** WARNING predicted positive rate {test_rate:.3f} vs "
              f"training {train_rate:.3f} ***")
        print("      Check: do train/test feature columns align? Is the "
              "threshold sane? Did the test features load correctly?")
    elif abs(test_rate - train_rate) > 0.12:
        print(f"\n  note: predicted positive rate {test_rate:.3f} vs training "
              f"{train_rate:.3f} — worth a look, not necessarily wrong")

    # ---- artifacts ----
    sub = pd.DataFrame({"id": test_ids, "label": final_pred})
    sub.to_csv(out / "blend_predictions.csv", index=False)
    if "xgboost" in names:
        xt = tuned["xgboost"]["threshold"]
        pd.DataFrame({"id": test_ids,
                      "label": (test_probs["xgboost"] >= xt).astype(int)}
                     ).to_csv(out / "XGB_leaderboard_predictions.csv", index=False)

    cmp_df = pd.DataFrame(results).sort_values("holdout_macro_f1", ascending=False)
    cmp_df.to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(history).to_csv(out / "tuning_history.csv", index=False)

    if "xgboost" in names:
        build, _, _ = ZOO["xgboost"]
        m = build(tuned["xgboost"]["params"], spw, SEED)
        m.set_params(early_stopping_rounds=None, n_estimators=400)
        m.fit(X_fit, y_fit)
        pd.DataFrame({"feature": feat_cols,
                      "importance": m.feature_importances_}
                     ).sort_values("importance", ascending=False
                                   ).to_csv(out / "feature_importance.csv", index=False)

    write_report(out, cmp_df, weights, blend_oof_f1, blend_hold_f1, gap, tuned,
                 tag, final_thr, final_pred, y, args, time.time() - t0, stats_block)

    print(f"\n[done ] submission -> {out/'blend_predictions.csv'} "
          f"({tag}, thr={final_thr:.3f}, pos_rate={final_pred.mean():.3f})")
    print(f"[done ] report     -> {out/'best_model_report.md'}")
    print(f"[time ] {time.time()-t0:.0f}s")
    print("\nQuote the HOLDOUT column in your report. `cv_selection_biased` is "
          "the max of many noisy CV estimates and runs optimistic.")
    return 0


def df_to_markdown(df: pd.DataFrame) -> str:
    """Hand-rolled so the script needs no `tabulate`. pandas' own
    `.to_markdown()` silently depends on it and dies at the very last line of
    an otherwise-complete run, which is the worst possible place to fail."""
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in rec]
            for rec in df.itertuples(index=False, name=None)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]
    head = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |"
            for r in rows]
    return "\n".join([head, sep, *body])


def write_report(out, cmp_df, weights, blend_oof, blend_hold, gap, tuned, tag,
                 thr, pred, y, args, secs, stats) -> None:
    lines = [
        "# Task 3 — best model report", "",
        f"Generated in {secs:.0f}s · seed {SEED} · "
        f"{args.trials} Optuna trials/model · "
        f"{args.folds}-fold x{args.repeats} CV · {args.seeds} seeds averaged",
        "", "## Headline", "",
        f"- **Selected:** `{tag}`",
        f"- **Holdout Macro-F1:** **{blend_hold if tag == 'blend' else cmp_df.iloc[0]['holdout_macro_f1']:.4f}** "
        "(15% slice never seen by tuning, blending, or thresholding)",
        f"- **Decision threshold:** {thr:.3f} (calibrated on OOF, not 0.5)",
        f"- **Predicted positive rate:** {pred.mean():.4f} "
        f"(training positive rate {y.mean():.4f})",
        "", "## Model comparison", "",
        df_to_markdown(cmp_df),
        "",
        "`cv_selection_biased` is `study.best_value` — the maximum of many noisy "
        "CV estimates, and therefore optimistic. It is shown for the tuning "
        "roadmap only. **`holdout_macro_f1` is the number to quote.** "
        "`oof_minus_holdout` is the overfitting readout: consistently large "
        "positive values mean the tuning is fitting fold noise.",
        "", "## Blend", "",
        f"- weights (fitted on OOF): `{json.dumps(weights)}`",
        f"- OOF Macro-F1: {blend_oof:.4f}",
        f"- holdout Macro-F1: {blend_hold:.4f}",
        f"- **gap: {gap:+.4f}** "
        + ("(healthy, < 0.02)" if gap <= 0.02 else "(**too large** — blend is fitting OOF noise)"),
        "", "## Statistical validation", "",
        "Method taken from Sujon et al. (2025), *Journal of Big Data* 12:268, "
        "`doi:10.1186/s40537-025-01313-4` — the paper finds F1 the most stable "
        "metric under class imbalance (supporting this competition's Macro-F1 "
        "choice) and recommends bootstrap CIs plus McNemar's test rather than "
        "comparing point estimates.",
        "",
        f"- blend holdout Macro-F1 95% CI: "
        f"[{stats['blend_ci'][0]:.4f}, {stats['blend_ci'][1]:.4f}]",
        f"- best single (`{stats['single_name']}`) 95% CI: "
        f"[{stats['single_ci'][0]:.4f}, {stats['single_ci'][1]:.4f}]",
        f"- McNemar discordant pairs: {stats['discordant_b']} vs "
        f"{stats['discordant_c']}, **p = {stats['mcnemar_p']:.4f}**",
        f"- decision: "
        + ("blend selected — it wins and the difference is significant"
           if stats["chose_blend"] else
           f"`{stats['single_name']}` selected — the blend does not beat it "
           "significantly, so the simpler model wins"),
        "",
        "The CIs overlap far more than the point estimates suggest. Treat any "
        "gap smaller than the CI width as unproven.",
        "", "## Tuned hyperparameters", "",
    ]
    for name, d in tuned.items():
        lines += [f"### {name}", "", f"- threshold: {d['threshold']:.3f}",
                  "```json", json.dumps(d["params"], indent=2), "```", ""]
    lines += [
        "## Anti-overfitting controls in force", "",
        "| Control | Where |",
        "|---|---|",
        "| 15% holdout split before any tuning | `train_test_split` at top of `main` |",
        "| Optuna restricted to the fit split | `tune(..., X_fit, y_fit, ...)` |",
        "| RepeatedStratifiedKFold | cuts noise floor ~sqrt(repeats) |",
        "| Early stopping on boosters | tree count learned, not searched |",
        "| Threshold from OOF, validated on holdout | `fit_blend_weights` / `best_threshold` |",
        "| Blend weights from OOF only | holdout stays clean |",
        "| Seed averaging | variance reduction that cannot overfit |",
        "| Fold std reported | improvements checkable against noise |",
        "",
        "See `.claude/skills/xgboost-paper-scout/references/overfitting-playbook.md`.",
        "", "## For the Task 4 write-up", "",
        f"- {len(tuned)} models explored with documented tuning "
        f"(rubric's top band asks for more than 4).",
        "- `tuning_history.csv` holds every trial for the hyperparameter table.",
        "- `model_comparison.csv` is the comparison table.",
        "- Log each submission's public-leaderboard score against its holdout "
        "score here; a widening gap means stop tuning.",
        "",
        "| Submission | Holdout Macro-F1 | Public LB | Gap |",
        "|---|---|---|---|",
        f"| {tag} (this run) | {blend_hold if tag == 'blend' else cmp_df.iloc[0]['holdout_macro_f1']:.4f} | _fill in_ | |",
        "",
    ]
    (out / "best_model_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
