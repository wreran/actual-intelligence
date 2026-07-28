"""
Task 1: Logistic Regression from Scratch
50.007 Machine Learning - GenAI Content Detection Project
NOT ALLOWED to use sklearn.linear_model.LogisticRegression or any predefined LR package.
Implements: sigmoid, cross-entropy loss, gradient descent (batch/mini-batch), L2 regularization.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

# Paths resolve from this file, not the shell's cwd.
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iters=2000, l2=0.0, batch_size=None, verbose=False, random_state=42):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _init_params(self, n_features):
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(0, 0.01, n_features)
        self.bias = 0.0

    def _compute_loss(self, y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        ce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        reg = (self.l2 / (2 * len(y_true))) * np.sum(self.weights ** 2)
        return ce + reg

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self._init_params(n_features)
        bs = self.batch_size or n_samples
        rng = np.random.default_rng(self.random_state)

        for it in range(self.n_iters):
            idx = rng.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]
            for start in range(0, n_samples, bs):
                end = start + bs
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]
                m = X_batch.shape[0]

                linear = X_batch @ self.weights + self.bias
                y_pred = self._sigmoid(linear)

                grad_w = (X_batch.T @ (y_pred - y_batch)) / m + (self.l2 / m) * self.weights
                grad_b = np.mean(y_pred - y_batch)

                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b

            if it % max(1, self.n_iters // 20) == 0 or it == self.n_iters - 1:
                full_pred = self._sigmoid(X @ self.weights + self.bias)
                loss = self._compute_loss(y, full_pred)
                self.loss_history.append((it, loss))
                if self.verbose:
                    print(f"Iter {it:5d} | loss={loss:.6f}")
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def load_data(train_features_path=DATA / "train_features.csv", test_features_path=DATA / "test_features.csv",
              train_labels_path=DATA / "train.csv"):
    """
    Adjust column names below to match your actual course-provided CSVs.
    train_features.csv / test_features.csv: MUST-use preprocessed TF-IDF-like features (Task 1&2 requirement).
    train.csv: original text + label column (assumed 'label' or 'Label_A' -- change as needed).
    """
    X_train = pd.read_csv(train_features_path)
    X_test = pd.read_csv(test_features_path)
    train_raw = pd.read_csv(train_labels_path)

    label_col_candidates = ["label", "Label_A", "target", "y"]
    label_col = next((c for c in label_col_candidates if c in train_raw.columns), None)
    if label_col is None:
        raise ValueError(f"Could not find label column in train.csv. Columns found: {train_raw.columns.tolist()}")
    y_train = train_raw[label_col].values

    id_cols = [c for c in X_train.columns if c.lower() in ("id", "index")]
    if id_cols:
        test_ids = X_test[id_cols[0]].values
        X_train = X_train.drop(columns=id_cols)
        X_test = X_test.drop(columns=id_cols)
    else:
        test_ids = np.arange(len(X_test))

    return X_train.values.astype(np.float64), y_train.astype(np.float64), X_test.values.astype(np.float64), test_ids


def train_val_split(X, y, val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


if __name__ == "__main__":
    X_train_full, y_train_full, X_test, test_ids = load_data()

    X_train, y_train, X_val, y_val = train_val_split(X_train_full, y_train_full, val_frac=0.1)

    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mu) / sigma
    X_val_n = (X_val - mu) / sigma
    X_test_n = (X_test - mu) / sigma

    model = LogisticRegressionScratch(lr=0.5, n_iters=1000, l2=0.01, batch_size=256, verbose=True)
    model.fit(X_train_n, y_train)

    val_pred = model.predict(X_val_n)
    print("Validation Accuracy:", accuracy_score(y_val, val_pred))
    print("Validation Macro-F1:", f1_score(y_val, val_pred, average="macro"))

    test_pred = model.predict(X_test_n)
    out = pd.DataFrame({"id": test_ids, "label": test_pred})
    out.to_csv(OUT / "LogReg_predictions.csv", index=False)
    print(f"Saved {OUT / 'LogReg_predictions.csv'}")
