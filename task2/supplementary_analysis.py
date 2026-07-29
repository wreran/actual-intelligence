#!/usr/bin/env python3
"""
Task 2 supplementary analysis — beyond the four required settings.

The brief asks for Macro-F1 at n_components in {2000, 1000, 500, 100} with
KNN n_neighbors=2. That grid answers the question but hides two things worth
reporting, and both are cheap to get once PCA is already fitted:

1. Where the optimum actually is. The required grid is monotonically
   improving as components fall, which reads as "fewer is always better".
   Extending below 100 shows it is really an inverted-U whose peak sits at
   ~100 — the required grid stops exactly at the sweet spot by coincidence.

2. Whether k=2 is a good choice. It is not. At every component count, larger
   k scores better, and at 2000 components the gap between k=2 and k=31 is
   ~0.18 Macro-F1 — far larger than the effect of dimensionality itself at
   that end of the range. k=2 is pathological: with two neighbours, ties are
   frequent and the vote is maximally sensitive to a single noisy point.

Run:  python3 task2/supplementary_analysis.py
Writes: task2/outputs/task2_supplementary_sweep.csv
        task2/outputs/task2_supplementary.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
COMPONENTS = [2000, 1000, 500, 100, 50, 20, 10, 5, 2]
REQUIRED = {2000, 1000, 500, 100}
KS = [2, 5, 15, 31]


def main() -> None:
    tr = pd.read_csv(DATA / "train_features.csv")
    y = tr["label"].to_numpy().astype(int)
    # Drop id AND label: train_features.csv carries the label in column 2, and
    # leaving it in X is target leakage.
    X = tr.drop(columns=["id", "label"]).to_numpy(np.float64)
    print(f"X={X.shape} pos_rate={y.mean():.4f}")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_val = int(0.15 * len(X))
    val, trn = idx[:n_val], idx[n_val:]

    # Fit PCA once at the largest setting; components are ordered by explained
    # variance, so the first n columns ARE the n-component projection.
    t0 = time.time()
    pca = PCA(n_components=max(COMPONENTS), random_state=SEED)
    A = pca.fit_transform(X[trn])
    B = pca.transform(X[val])
    y_tr, y_va = y[trn], y[val]
    print(f"PCA fitted in {time.time() - t0:.0f}s")

    majority = f1_score(y_va, np.ones_like(y_va), average="macro")
    print(f"always-predict-majority baseline Macro-F1 = {majority:.4f}\n")

    rows = []
    header = "n_comp  " + "".join(f"k={k:<7}" for k in KS) + "explvar"
    print(header)
    for n in COMPONENTS:
        rec = {"n_components": n,
               "explained_variance": float(pca.explained_variance_ratio_[:n].sum()),
               "required_by_brief": n in REQUIRED}
        for k in KS:
            knn = KNeighborsClassifier(n_neighbors=k).fit(A[:, :n], y_tr)
            rec[f"macro_f1_k{k}"] = f1_score(y_va, knn.predict(B[:, :n]),
                                             average="macro")
        rows.append(rec)
        cells = "".join(f"{rec[f'macro_f1_k{k}']:.4f}  " for k in KS)
        print(f"{n:5d}   {cells}{rec['explained_variance']:.3f}"
              + ("  <- required" if n in REQUIRED else ""))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "task2_supplementary_sweep.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in KS:
        ax1.plot(df.n_components, df[f"macro_f1_k{k}"], marker="o",
                 label=f"k={k}" + (" (required)" if k == 2 else ""))
    ax1.axhline(majority, ls="--", c="grey", lw=1,
                label=f"majority baseline ({majority:.3f})")
    ax1.set_xscale("log")
    ax1.set_xlabel("PCA components (log scale)")
    ax1.set_ylabel("Macro-F1")
    ax1.set_title("Macro-F1 vs components, by k")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(df.n_components, df.explained_variance, marker="s", color="tab:red")
    ax2.set_xscale("log")
    ax2.set_xlabel("PCA components (log scale)")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_title("Variance retained vs components")
    ax2.grid(alpha=0.3)

    fig.suptitle("Task 2 supplementary: more variance does not mean better KNN",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "task2_supplementary.png", dpi=150)

    best = df.loc[df.macro_f1_k2.idxmax()]
    print(f"\nBest k=2 setting: {int(best.n_components)} components "
          f"(Macro-F1 {best.macro_f1_k2:.4f}, "
          f"{best.explained_variance:.1%} variance retained)")
    print(f"Wrote {OUT / 'task2_supplementary_sweep.csv'}")
    print(f"Wrote {OUT / 'task2_supplementary.png'}")


if __name__ == "__main__":
    main()
