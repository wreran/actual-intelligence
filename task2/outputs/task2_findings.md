# Task 2 — PCA + KNN results

Seed 42. Fitted on 85% of `train_features.csv` (17,000 rows), scored on a
held-out 15% validation split (3,000 rows).

**On "report Macro-F1 on the test set":** the Kaggle test labels are hidden, so
Macro-F1 cannot be computed there. These numbers come from a held-out
validation split carved out of the training set — the standard substitute.
Say so explicitly in the report.

## Required table

KNN with `n_neighbors=2`, as mandated.

| n_components | Macro-F1 | Cumulative explained variance |
|---|---|---|
| 2000 | 0.4259 | 77.8% |
| 1000 | 0.4962 | 57.2% |
| 500  | 0.5733 | 39.5% |
| **100**  | **0.6511** | 15.9% |

Always-predict-majority baseline: **0.3812**. At 2000 components KNN scores
0.4259 — barely above predicting "AI" for everything.

Submission written from the best setting (100 components):
`PCA_KNN_predictions.csv`.

## The trend, and why

**Macro-F1 rises as components fall, while explained variance collapses.**
Going from 2000 to 100 components throws away 62 percentage points of variance
and *gains* 0.225 Macro-F1. Retaining variance and retaining useful signal are
not the same thing.

The mechanism is distance concentration. KNN classifies by Euclidean distance,
and in high dimensions the ratio between the nearest and farthest neighbour
tends to 1 — every point becomes roughly equidistant from every other, so
"nearest" stops meaning "similar". With 2000 TF-IDF-derived dimensions the
neighbourhood is essentially arbitrary. Cutting to 100 dimensions concentrates
the retained variance into directions that still separate the classes while
restoring meaningful distances.

Note also that the trailing principal components of a TF-IDF matrix are mostly
noise directions from rare terms. Discarding them removes noise, not signal —
which is why the variance loss costs nothing.

## Two things the required grid hides

`supplementary_analysis.py` extends the sweep. See
`task2_supplementary_sweep.csv` and `task2_supplementary.png`.

### 1. The optimum sits at 100 — the grid stops exactly at the peak

| n_components | 2000 | 1000 | 500 | **100** | 50 | 20 | 10 | 5 | 2 |
|---|---|---|---|---|---|---|---|---|---|
| Macro-F1 (k=2) | 0.4259 | 0.4962 | 0.5733 | **0.6511** | 0.6457 | 0.6400 | 0.6141 | 0.5907 | 0.5368 |

The relationship is not monotonic — it is an inverted U. Below 100 components
performance falls again, down to 0.5368 at 2 components, because too much
class-discriminating signal has been discarded. The required grid happens to
terminate precisely at the maximum, which makes the four mandated points look
like a monotonic trend when they are really the left arm of a curve.

This is the bias-variance tradeoff in dimensionality: too many dimensions and
distances are meaningless (variance), too few and the classes are no longer
separable (bias).

### 2. `n_neighbors=2` is the worst choice of k tested, at every setting

| n_components | k=2 | k=5 | k=15 | k=31 |
|---|---|---|---|---|
| 2000 | 0.4259 | 0.4646 | 0.4962 | **0.6060** |
| 1000 | 0.4962 | 0.5674 | 0.5722 | **0.5961** |
| 500  | 0.5733 | **0.6277** | 0.6205 | 0.6201 |
| 100  | 0.6511 | 0.6700 | 0.6789 | **0.6821** |

At 2000 components, moving from k=2 to k=31 is worth **+0.18 Macro-F1** — a
larger effect than dimensionality reduction produces anywhere in that region.

k=2 is a pathological choice for a binary problem. With two neighbours, a
one-vote-each split is common and must be broken arbitrarily (sklearn falls
back to the lower class index), and a single mislabelled or atypical training
point flips half the vote. Larger odd k averages over more evidence and cannot
tie. The brief mandates k=2, so the required table uses it — but the
sensitivity is worth reporting as a limitation of the prescribed setup rather
than of KNN itself.

## For the write-up

- PCA is fitted **once** at 2000 components and sliced for the smaller
  settings. Principal components are ordered by explained variance, so the
  first n columns of a 2000-component projection are exactly the n-component
  projection. Fitting four separate PCAs repeats the same SVD for identical
  numbers.
- PCA is fitted on the training split only and applied to validation/test via
  `transform`. Fitting it on all the data first would leak the validation set's
  covariance structure into the projection.
- Even at its best (0.6511), PCA+KNN sits well below the Task 3 models
  (~0.75). The interesting content here is the *why*, not the score.
