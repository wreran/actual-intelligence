"""
Task 3: Compete in Leaderboard - XGBoost with engineered features
50.007 Machine Learning - GenAI Content Detection Project
Constraint: DO NOT use deep learning or LLMs as the classifier itself.
Uses XGBoost as primary model, with feature engineering informed by:
 - NELA-style stylometric/complexity features (Malviya et al., SKDU De-Factify 2025)
 - DivEye surprisal statistics (Basani & Chen, 2026) [OPTIONAL - requires a small frozen LM
   for feature extraction only, not as the classifier; disclose if used, per course rules]
 - Curvature-inspired / readability cues (NOTAI.AI, 2026)
"""
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import optuna

# Paths resolve from this file, not the shell's cwd.
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Stylometric / complexity feature engineering (cheap, no LM required)
# ---------------------------------------------------------------------------

def extract_stylometric_features(text: str) -> dict:
    text = str(text)
    words = re.findall(r"\b\w+\b", text.lower())
    n_words = len(words) or 1
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    n_sentences = len(sentences) or 1
    unique_words = set(words)
    punct_counts = Counter(c for c in text if c in string.punctuation)
    stopwords = {"the","a","an","is","are","was","were","in","on","at","of","to","and","or","but","it","this","that"}
    stopword_count = sum(1 for w in words if w in stopwords)
    word_freq = Counter(words)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)

    return {
        "char_count": len(text),
        "word_count": n_words,
        "sentence_count": n_sentences,
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "avg_sentence_len": n_words / n_sentences,
        "type_token_ratio": len(unique_words) / n_words,
        "hapax_ratio": hapax / n_words,
        "stopword_ratio": stopword_count / n_words,
        "punct_count": sum(punct_counts.values()),
        "comma_count": punct_counts.get(",", 0),
        "period_count": punct_counts.get(".", 0),
        "exclaim_count": punct_counts.get("!", 0),
        "question_count": punct_counts.get("?", 0),
        "digit_ratio": sum(c.isdigit() for c in text) / max(len(text), 1),
        "upper_ratio": sum(c.isupper() for c in text) / max(len(text), 1),
    }


def build_stylometric_matrix(texts: pd.Series) -> pd.DataFrame:
    feats = texts.apply(extract_stylometric_features).tolist()
    return pd.DataFrame(feats)


# ---------------------------------------------------------------------------
# 2. (Optional, disclose in report) Surprisal features inspired by DivEye
#    Requires: pip install torch transformers
#    NOTE: Using a frozen LM purely for FEATURE EXTRACTION (not as the
#    classifier) is used here only to compute statistical features;
#    the actual classifier remains XGBoost. If your course interprets
#    "no deep learning" as banning ANY neural network touch, SKIP this
#    block and rely on stylometric features only -- clear this with your TA.
# ---------------------------------------------------------------------------

def extract_diveye_features(texts, model_name="gpt2", device="cpu", max_tokens=256):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from scipy.stats import skew, kurtosis

    tok = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for text in texts:
            ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.to(device)
            if ids.shape[1] < 3:
                rows.append([0.0] * 9)
                continue
            out = model(ids)
            logits = out.logits[0, :-1, :]
            targets = ids[0, 1:]
            logprobs = torch.log_softmax(logits, dim=-1)
            surprisal = -logprobs[torch.arange(len(targets)), targets].cpu().numpy()

            mu = surprisal.mean()
            var = surprisal.var(ddof=1) if len(surprisal) > 1 else 0.0
            sk = skew(surprisal) if len(surprisal) > 2 else 0.0
            ku = kurtosis(surprisal) if len(surprisal) > 2 else 0.0

            d1 = np.diff(surprisal)
            d1_mean = d1.mean() if len(d1) else 0.0
            d1_var = d1.var(ddof=1) if len(d1) > 1 else 0.0

            d2 = np.diff(d1)
            d2_var = d2.var(ddof=1) if len(d2) > 1 else 0.0
            if len(d2) > 1:
                hist, _ = np.histogram(d2, bins=20, density=True)
                p = hist / (hist.sum() + 1e-12)
                entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
                if len(d2) > 2 and d2_var > 0:
                    autocorr = np.corrcoef(d2[:-1], d2[1:])[0, 1]
                else:
                    autocorr = 0.0
            else:
                entropy, autocorr = 0.0, 0.0

            rows.append([mu, var, sk, ku, d1_mean, d1_var, d2_var, entropy, autocorr])

    cols = ["surp_mean","surp_var","surp_skew","surp_kurt",
            "surp_d1_mean","surp_d1_var","surp_d2_var","surp_d2_entropy","surp_d2_autocorr"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# 3. Data loading and feature assembly
# ---------------------------------------------------------------------------

def load_and_build_features(train_features_path=DATA / "train_features.csv",
                             test_features_path=DATA / "test_features.csv",
                             train_raw_path=DATA / "train.csv",
                             test_raw_path=DATA / "test.csv",
                             text_col="text",
                             use_diveye=False):
    X_train_feat = pd.read_csv(train_features_path)
    X_test_feat = pd.read_csv(test_features_path)
    train_raw = pd.read_csv(train_raw_path)
    test_raw = pd.read_csv(test_raw_path)

    label_col_candidates = ["label", "Label_A", "target", "y"]
    label_col = next((c for c in label_col_candidates if c in train_raw.columns), None)
    y_train = train_raw[label_col].values

    id_cols = [c for c in X_train_feat.columns if c.lower() in ("id", "index")]
    test_ids = X_test_feat[id_cols[0]].values if id_cols else np.arange(len(X_test_feat))
    if id_cols:
        X_train_feat = X_train_feat.drop(columns=id_cols)
        X_test_feat = X_test_feat.drop(columns=id_cols)

    train_style = build_stylometric_matrix(train_raw[text_col])
    test_style = build_stylometric_matrix(test_raw[text_col])

    X_train_full = pd.concat([X_train_feat.reset_index(drop=True), train_style.reset_index(drop=True)], axis=1)
    X_test_full = pd.concat([X_test_feat.reset_index(drop=True), test_style.reset_index(drop=True)], axis=1)

    if use_diveye:
        train_div = extract_diveye_features(train_raw[text_col].tolist())
        test_div = extract_diveye_features(test_raw[text_col].tolist())
        X_train_full = pd.concat([X_train_full, train_div], axis=1)
        X_test_full = pd.concat([X_test_full, test_div], axis=1)

    return X_train_full, y_train, X_test_full, test_ids


# ---------------------------------------------------------------------------
# 4. Optuna hyperparameter search optimizing Macro-F1 via Stratified K-Fold
# ---------------------------------------------------------------------------

def macro_f1_cv_score(params, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, average="macro"))
    return float(np.mean(scores))


def build_optuna_objective(X, y, n_splits=5, seed=42):
    neg, pos = np.bincount(y.astype(int))
    scale_pos_weight = neg / max(pos, 1)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        return macro_f1_cv_score(params, X, y, n_splits=n_splits, seed=seed)

    return objective


def run_optuna_search(X, y, n_trials=50, n_splits=5, seed=42):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(build_optuna_objective(X, y, n_splits=n_splits, seed=seed), n_trials=n_trials)
    print("Best Macro-F1 (CV):", study.best_value)
    print("Best params:", study.best_params)
    return study


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    USE_DIVEYE = False  # set True only after confirming with TA that LM-based feature extraction is allowed

    X_train_df, y_train, X_test_df, test_ids = load_and_build_features(use_diveye=USE_DIVEYE)
    X_train_df = X_train_df.fillna(0)
    X_test_df = X_test_df.fillna(0)

    X_train = X_train_df.values.astype(np.float64)
    X_test = X_test_df.values.astype(np.float64)
    y_train = np.asarray(y_train).astype(int)

    study = run_optuna_search(X_train, y_train, n_trials=50, n_splits=5)

    neg, pos = np.bincount(y_train)
    best_params = dict(study.best_params)
    best_params.update({
        "scale_pos_weight": neg / max(pos, 1),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    })

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    importances = pd.Series(final_model.feature_importances_, index=X_train_df.columns).sort_values(ascending=False)
    importances.to_csv(OUT / "task3_feature_importance.csv")
    print(importances.head(20))

    test_preds = final_model.predict(X_test)
    pd.DataFrame({"id": test_ids, "label": test_preds}).to_csv(OUT / "XGB_leaderboard_predictions.csv", index=False)
    print(f"Saved {OUT / 'XGB_leaderboard_predictions.csv'}")

    pd.Series(best_params).to_csv(OUT / "task3_best_hyperparameters.csv")
    print(f"Saved best hyperparameters to {OUT}")
