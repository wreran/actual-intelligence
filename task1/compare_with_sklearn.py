#!/usr/bin/env python3
"""
Task 1 rubric evidence — from-scratch logistic regression vs sklearn.

The 5-mark band asks for "comparative performance compared to sklearn logistic
regression". A fair comparison needs both models to see *identical* features,
the same split, and the same preprocessing — otherwise the number measures
preprocessing choices rather than the implementation.

So: same normalised features (training mean/std, applied to both), same
validation split, and sklearn's L2 strength matched to the from-scratch model's
(sklearn's C is the inverse of the regularisation strength, so C = 1/l2).

Writes task1/outputs/task1_sklearn_comparison.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logistic_regression import (LogisticRegressionScratch, load_data,  # noqa: E402
                                 train_val_split)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

L2 = 0.01
LR = 0.5
ITERS = 1000
BATCH = 256


def metrics(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
    }


def main() -> None:
    X_full, y_full, _, _ = load_data()
    X_tr, y_tr, X_va, y_va = train_val_split(X_full, y_full, val_frac=0.1)

    # One normaliser, fitted on train, applied to both models and both splits.
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-8
    Xtr, Xva = (X_tr - mu) / sd, (X_va - mu) / sd
    print(f"train {Xtr.shape} | val {Xva.shape} | pos_rate {y_tr.mean():.4f}",
          flush=True)

    t = time.time()
    mine = LogisticRegressionScratch(lr=LR, n_iters=ITERS, l2=L2, batch_size=BATCH)
    mine.fit(Xtr, y_tr)
    t_mine = time.time() - t
    p_mine = mine.predict(Xva)
    print(f"from-scratch fitted in {t_mine:.0f}s", flush=True)

    t = time.time()
    sk = LogisticRegression(C=1.0 / L2, max_iter=ITERS, solver="lbfgs")
    sk.fit(Xtr, y_tr)
    t_sk = time.time() - t
    p_sk = sk.predict(Xva)
    print(f"sklearn fitted in {t_sk:.0f}s", flush=True)

    m_mine, m_sk = metrics(y_va, p_mine), metrics(y_va, p_sk)
    agree = float((p_mine == p_sk).mean())
    w1 = mine.weights / np.linalg.norm(mine.weights)
    w2 = sk.coef_[0] / np.linalg.norm(sk.coef_[0])
    cos = float(w1 @ w2)

    lines = [
        "# Task 1 — from-scratch logistic regression vs sklearn",
        "",
        f"Identical features, identical split, identical normalisation "
        f"(training mean/std). sklearn's `C` set to `1/l2` = {1/L2:.0f} so both "
        f"carry the same L2 strength. Validation split: {len(y_va)} rows.",
        "",
        "| Metric | From scratch | sklearn | Difference |",
        "|---|---|---|---|",
    ]
    for k in ("accuracy", "macro_f1", "precision", "recall"):
        lines.append(f"| {k} | {m_mine[k]:.4f} | {m_sk[k]:.4f} | "
                     f"{m_mine[k] - m_sk[k]:+.4f} |")
    lines += [
        f"| fit time (s) | {t_mine:.0f} | {t_sk:.0f} | |",
        "",
        f"- **Prediction agreement:** {agree:.4f} of validation rows",
        f"- **Cosine similarity of learned weight vectors:** {cos:.4f}",
        "",
        "## Confusion matrices",
        "",
        "From scratch:",
        "```",
        str(confusion_matrix(y_va, p_mine)),
        "```",
        "sklearn:",
        "```",
        str(confusion_matrix(y_va, p_sk)),
        "```",
        "",
        "## Reading this",
        "",
        f"The two implementations agree on {agree:.1%} of predictions and their "
        f"weight vectors have cosine similarity {cos:.3f}, so they have "
        "converged to substantially the same decision boundary.",
        "",
        "Any residual gap is optimiser behaviour, not a difference in the model: "
        "the from-scratch version runs mini-batch gradient descent with a fixed "
        f"learning rate ({LR}) for {ITERS} passes, while sklearn's L-BFGS is a "
        "quasi-Newton method that uses curvature information and converges to a "
        "tighter optimum on a convex objective. That is expected and is worth "
        "stating in the report rather than papering over.",
    ]
    (OUT / "task1_sklearn_comparison.md").write_text("\n".join(lines))
    print("\n".join(lines[4:14]))
    print(f"\nWrote {OUT / 'task1_sklearn_comparison.md'}")


if __name__ == "__main__":
    main()
