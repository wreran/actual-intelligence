"""
Task 2: PCA + KNN
50.007 Machine Learning - GenAI Content Detection Project
sklearn IS allowed for this task.
Requirement: report Macro-F1 for n_components in {2000, 1000, 500, 100}, KNN with n_neighbors=2.
"""
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

from pathlib import Path

# Paths resolve from this file, not the shell's cwd, so the script runs the same
# from the repo root or from inside task2/.
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def load_data(train_features_path=DATA / "train_features.csv",
              test_features_path=DATA / "test_features.csv",
              train_labels_path=DATA / "train.csv", test_labels_path=None):
    X_train = pd.read_csv(train_features_path)
    X_test = pd.read_csv(test_features_path)
    train_raw = pd.read_csv(train_labels_path)

    label_col_candidates = ["label", "Label_A", "target", "y"]
    label_col = next((c for c in label_col_candidates if c in train_raw.columns), None)
    y_train = train_raw[label_col].values

    id_cols = [c for c in X_train.columns if c.lower() in ("id", "index")]
    test_ids = X_test[id_cols[0]].values if id_cols else np.arange(len(X_test))

    # train_features.csv is [id, label, 0001..5000] — it carries the LABEL as
    # well as the id. Dropping only the id columns leaves `label` sitting in X
    # as a feature, which is target leakage: any model then scores ~1.0 on
    # validation and collapses on the leaderboard. Here it surfaced as a shape
    # mismatch (5001 train vs 5000 test features) only by luck.
    #
    # Take the feature columns as the intersection with the test set, which
    # cannot contain the label by construction.
    feature_cols = [c for c in X_train.columns
                    if c in set(X_test.columns) and c not in id_cols]
    dropped = [c for c in X_train.columns if c not in feature_cols and c not in id_cols]
    if dropped:
        print(f"  dropped non-feature columns from X: {dropped}")
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]
    print(f"  feature matrix: train {X_train.shape} test {X_test.shape}")

    y_test = None
    if test_labels_path is not None:
        test_raw = pd.read_csv(test_labels_path)
        label_col_t = next((c for c in label_col_candidates if c in test_raw.columns), None)
        if label_col_t:
            y_test = test_raw[label_col_t].values

    return X_train.values.astype(np.float64), y_train, X_test.values.astype(np.float64), test_ids, y_test


def run_pca_knn_sweep(X_train, y_train, X_eval, y_eval, component_list=(2000, 1000, 500, 100), n_neighbors=2, seed=42):
    """
    Returns dict {n_components: {"f1": macro_f1, "pca": pca, "knn": knn, ...}}.

    PCA is fitted ONCE at the largest requested component count, then sliced.
    Principal components come out ordered by explained variance, so the first
    n columns of a 2000-component projection are exactly the n-component
    projection — fitting PCA separately per setting would repeat the same
    expensive SVD four times for identical numbers.

    Note on "on the test set": the Kaggle test labels are hidden, so Macro-F1
    cannot be computed there directly. This sweep therefore scores a held-out
    validation split carved from the training set, which is the standard
    substitute. State that in the report.
    """
    results = {}
    max_components = min(X_train.shape[0], X_train.shape[1], max(component_list))
    print(f"Fitting PCA once at n_components={max_components} "
          f"(then slicing for the smaller settings) ...")
    t0 = time.time()
    pca = PCA(n_components=max_components, random_state=seed)
    X_train_full = pca.fit_transform(X_train)
    X_eval_full = pca.transform(X_eval)
    print(f"  PCA fitted in {time.time() - t0:.0f}s | "
          f"total explained variance = {pca.explained_variance_ratio_.sum():.4f}")

    for n_comp in component_list:
        n_eff = min(n_comp, max_components)
        t1 = time.time()
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train_full[:, :n_eff], y_train)
        preds = knn.predict(X_eval_full[:, :n_eff])

        f1 = f1_score(y_eval, preds, average="macro") if y_eval is not None else None
        ev = float(pca.explained_variance_ratio_[:n_eff].sum())
        results[n_comp] = {"f1": f1, "pca": pca, "knn": knn, "n_eff": n_eff,
                           "explained_var": ev}
        print(f"n_components={n_eff:5d} | Macro-F1={f1:.4f} | "
              f"explained_var={ev:.4f} | knn {time.time() - t1:.0f}s")
    return results


def plot_f1_vs_components(results, save_path=OUT / "task2_f1_vs_components.png"):
    comps = [c for c in results if results[c]["f1"] is not None]
    f1s = [results[c]["f1"] for c in comps]
    if not comps:
        print("No F1 values available to plot (y_eval was None).")
        return
    order = np.argsort(comps)[::-1]
    comps = [comps[i] for i in order]
    f1s = [f1s[i] for i in order]
    plt.figure(figsize=(6, 4))
    plt.plot([str(c) for c in comps], f1s, marker="o")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Macro-F1 Score")
    plt.title("KNN (k=2) Macro-F1 vs PCA Components")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    X_train_full, y_train_full, X_test, test_ids, y_test_maybe = load_data()

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_train_full))
    n_val = int(0.15 * len(X_train_full))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_tr, y_tr = X_train_full[tr_idx], y_train_full[tr_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    results = run_pca_knn_sweep(X_tr, y_tr, X_val, y_val,
                                 component_list=(2000, 1000, 500, 100), n_neighbors=2)

    report_rows = [{"n_components": c, "macro_f1": results[c]["f1"],
                     "explained_variance": results[c]["explained_var"]} for c in results]
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(OUT / "task2_pca_knn_report_table.csv", index=False)
    print(report_df)

    plot_f1_vs_components(results)

    best_n = max(results, key=lambda c: (results[c]["f1"] or -1))
    best_pca, best_knn = results[best_n]["pca"], results[best_n]["knn"]
    # PCA was fitted once at the max component count; slice to the winner's
    # width so the KNN sees the same dimensionality it was fitted on.
    n_eff = results[best_n]["n_eff"]
    X_test_pca = best_pca.transform(X_test)[:, :n_eff]
    test_preds = best_knn.predict(X_test_pca)
    pd.DataFrame({"id": test_ids, "label": test_preds}).to_csv(OUT / "PCA_KNN_predictions.csv", index=False)
    print(f"Saved PCA_KNN_predictions.csv using n_components={best_n}")
