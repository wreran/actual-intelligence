"""
Hybrid AI-text detector — training pipeline.
============================================

Combines the verified best components of the three curated papers:

  1. SKDU/NELA (AAAI-25 De-Factify 4.0): length-normalized stylometric
     features -> the strongest *course-safe* signal (F1 0.9979 dev /
     0.9945 test with XGBoost on a near-identical binary task).
  2. DivEye (TMLR 2026): 9-dim surprisal-diversity vector (corrected
     composition: distribution 4 + first-order 2 + second-order
     var/entropy/autocorrelation) from a frozen GPT-2 — OPTIONAL, gated
     behind --with-lm since a frozen LM as feature extractor may need
     TA sign-off under a "no deep learning" rule.
  3. NotAI.AI / Fast-DetectGPT (ICLR 2024): analytic conditional
     probability curvature as an extra interpretable feature (also
     behind --with-lm), plus SHAP-based explanation of the final model.
  4. Kaggle/community consensus (practitioner folk wisdom, not a single
     canonical source): Optuna TPE over staged parameter groups
     (tree structure + regularization searched together, learning rate
     lowered with more trees at the end), StratifiedKFold CV scored on
     Macro-F1 (the competition metric), scale_pos_weight set from the
     class ratio (RAID-style imbalance insurance).
  5. MAGE (ACL 2024) OOD trick: after training, calibrate the decision
     threshold on a held-out slice by maximizing Macro-F1 instead of
     defaulting to 0.5 — cheap and recovers much OOD loss.

Usage
-----
  python train_hybrid.py --train train.csv --test test.csv \
      [--text-col text --label-col label] \
      [--with-lm] [--lm-name gpt2] [--trials 60] [--with-shap] \
      [--out-dir runs/hybrid]

Input CSVs need a text column and (train only) a binary label column
(0 = human, 1 = AI). Predictions are written to <out-dir>/predictions.csv.

Dependencies: numpy scipy pandas scikit-learn xgboost optuna
              (+ torch transformers for --with-lm, shap for --with-shap)
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from features_stylometric import extract_stylometric_matrix, try_nela_full

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 50007  # course number, for reproducibility jokes and rigor
N_SPLITS = 5


# --------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------

def build_features(texts: list[str], with_lm: bool, lm_name: str,
                   use_nela_pkg: bool, cache: Path | None = None,
                   tag: str = "") -> tuple[np.ndarray, list[str]]:
    """Assemble the hybrid feature matrix: stylometric [always] +
    original-NELA [if installed & requested] + DivEye/CPC [if --with-lm].

    Features are cached to <cache>/<tag>_features.npz because the LM pass
    is the slow part and Optuna will re-fit the booster many times.
    """
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / f"{tag}_features.npz"
        if f.exists():
            data = np.load(f, allow_pickle=True)
            print(f"[cache] loaded {tag} features from {f}")
            return data["X"], list(data["names"])

    X_sty, names = extract_stylometric_matrix(texts)
    print(f"[features] stylometric: {X_sty.shape[1]} dims")
    blocks, all_names = [X_sty], list(names)

    if use_nela_pkg:
        X_nela, nela_names = try_nela_full(texts)
        if X_nela is not None:
            blocks.append(X_nela)
            all_names += [f"nela_{n}" for n in nela_names]
            print(f"[features] original NELA package: {X_nela.shape[1]} dims")
        else:
            print("[features] nela_features not installed — skipping "
                  "(pip install nela_features to enable)")

    if with_lm:
        from features_surprisal import SurprisalExtractor
        ext = SurprisalExtractor(lm_name)
        X_lm, lm_names = ext.extract_matrix(texts)
        blocks.append(X_lm)
        all_names += lm_names
        print(f"[features] DivEye+CPC ({lm_name}): {X_lm.shape[1]} dims")

    X = np.hstack(blocks)
    if cache is not None:
        np.savez_compressed(cache / f"{tag}_features.npz",
                            X=X, names=np.array(all_names, dtype=object))
    return X, all_names


# --------------------------------------------------------------------------
# Optuna objective — Kaggle-consensus staged search space, Macro-F1 CV
# --------------------------------------------------------------------------

def make_objective(X: np.ndarray, y: np.ndarray, spw: float):
    import xgboost as xgb

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)

    def objective(trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "scale_pos_weight": spw,
            # --- tree structure (tune first, per community consensus) ---
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 0.5, 16.0, log=True),
            # --- sampling ---
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0),
            # --- regularization ---
            "gamma": trial.suggest_float("gamma", 1e-8, 4.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True),
            # --- learning rate / trees (coupled: low eta needs more trees) ---
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        }
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            model = xgb.XGBClassifier(**params)
            model.fit(X[tr_idx], y[tr_idx],
                      eval_set=[(X[va_idx], y[va_idx])], verbose=False)
            pred = model.predict(X[va_idx])
            # score the COMPETITION metric, not accuracy/logloss
            scores.append(f1_score(y[va_idx], pred, average="macro"))
        return float(np.mean(scores))

    return objective


# --------------------------------------------------------------------------
# MAGE-style threshold calibration
# --------------------------------------------------------------------------

def calibrate_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Pick the probability threshold maximizing Macro-F1 on held-out data
    (MAGE, ACL 2024: tiny in-domain calibration recovers most OOD loss)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        f1 = f1_score(y_true, (proba >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--with-lm", action="store_true",
                    help="add DivEye surprisal + Fast-DetectGPT CPC features "
                         "(frozen GPT-2; check course rules / TA sign-off)")
    ap.add_argument("--lm-name", default="gpt2")
    ap.add_argument("--use-nela-pkg", action="store_true",
                    help="also extract the original NELA toolkit vector "
                         "(requires pip install nela_features)")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--with-shap", action="store_true",
                    help="write SHAP global importance (NotAI.AI-style "
                         "explainability; requires pip install shap)")
    ap.add_argument("--out-dir", default="runs/hybrid")
    args = ap.parse_args()

    import optuna
    import xgboost as xgb

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    df = pd.read_csv(args.train)
    texts = df[args.text_col].astype(str).tolist()
    y = df[args.label_col].astype(int).to_numpy()
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    spw = n_neg / max(n_pos, 1)  # RAID-style imbalance insurance
    print(f"[data] {len(y)} rows | human={n_neg} ai={n_pos} "
          f"| scale_pos_weight={spw:.3f}")

    X, feat_names = build_features(texts, args.with_lm, args.lm_name,
                                   args.use_nela_pkg, cache=out, tag="train")

    # hold out a calibration slice BEFORE tuning (never touched by Optuna)
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE)

    # ---- Optuna TPE search, Macro-F1 objective ----
    print(f"[optuna] {args.trials} trials, {N_SPLITS}-fold stratified CV, "
          f"maximizing Macro-F1")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(make_objective(X_fit, y_fit, spw),
                   n_trials=args.trials, show_progress_bar=True)
    print(f"[optuna] best CV Macro-F1 = {study.best_value:.4f}")
    print(f"[optuna] best params: {study.best_params}")

    # ---- final fit on the full fit-split ----
    best = dict(study.best_params)
    best.update(objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=RANDOM_STATE,
                n_jobs=-1, scale_pos_weight=spw)
    model = xgb.XGBClassifier(**best)
    model.fit(X_fit, y_fit, verbose=False)

    # ---- MAGE-style threshold calibration on the untouched slice ----
    proba_cal = model.predict_proba(X_cal)[:, 1]
    thr = calibrate_threshold(y_cal, proba_cal)
    cal_f1 = f1_score(y_cal, (proba_cal >= thr).astype(int), average="macro")
    print(f"[calibration] threshold={thr:.3f} "
          f"(Macro-F1 on calibration slice: {cal_f1:.4f})")

    # refit on ALL training data with the tuned params before inference
    model_full = xgb.XGBClassifier(**best)
    model_full.fit(X, y, verbose=False)
    model_full.save_model(str(out / "model.json"))
    (out / "run_meta.json").write_text(json.dumps({
        "best_cv_macro_f1": study.best_value,
        "best_params": study.best_params,
        "threshold": thr,
        "calibration_macro_f1": cal_f1,
        "scale_pos_weight": spw,
        "n_features": len(feat_names),
        "with_lm": args.with_lm,
        "feature_names": feat_names,
    }, indent=2))

    # ---- NotAI.AI-style explainability ----
    if args.with_shap:
        try:
            import shap
            explainer = shap.TreeExplainer(model_full)
            sv = explainer.shap_values(X)
            imp = np.abs(sv).mean(axis=0)
            order = np.argsort(imp)[::-1]
            lines = [f"{feat_names[i]:28s} {imp[i]:.5f}" for i in order[:30]]
            (out / "shap_top30.txt").write_text("\n".join(lines))
            print("[shap] top-5 features:",
                  ", ".join(feat_names[i] for i in order[:5]))
        except ImportError:
            print("[shap] not installed — skipping (pip install shap)")

    # ---- inference ----
    if args.test:
        df_te = pd.read_csv(args.test)
        te_texts = df_te[args.text_col].astype(str).tolist()
        X_te, _ = build_features(te_texts, args.with_lm, args.lm_name,
                                 args.use_nela_pkg, cache=out, tag="test")
        proba = model_full.predict_proba(X_te)[:, 1]
        df_te["proba_ai"] = proba
        df_te["prediction"] = (proba >= thr).astype(int)
        df_te.to_csv(out / "predictions.csv", index=False)
        print(f"[done] predictions -> {out / 'predictions.csv'}")


if __name__ == "__main__":
    main()
