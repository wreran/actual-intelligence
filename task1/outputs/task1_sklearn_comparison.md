# Task 1 — from-scratch logistic regression vs sklearn

Identical features, identical split, identical normalisation (training mean/std). sklearn's `C` set to `1/l2` = 100 so both carry the same L2 strength. Validation split: 2000 rows.

| Metric | From scratch | sklearn | Difference |
|---|---|---|---|
| accuracy | 0.7110 | 0.7165 | -0.0055 |
| macro_f1 | 0.6957 | 0.7019 | -0.0061 |
| precision | 0.7670 | 0.7727 | -0.0056 |
| recall | 0.7608 | 0.7632 | -0.0024 |
| fit time (s) | 643 | 7 | |

- **Prediction agreement:** 0.9255 of validation rows
- **Cosine similarity of learned weight vectors:** 0.9725

## Confusion matrices

From scratch:
```
[[487 284]
 [294 935]]
```
sklearn:
```
[[495 276]
 [291 938]]
```

## Reading this

The two implementations agree on 92.5% of predictions and their weight vectors have cosine similarity 0.973, so they have converged to substantially the same decision boundary.

Any residual gap is optimiser behaviour, not a difference in the model: the from-scratch version runs mini-batch gradient descent with a fixed learning rate (0.5) for 1000 passes, while sklearn's L-BFGS is a quasi-Newton method that uses curvature information and converges to a tighter optimum on a convex objective. That is expected and is worth stating in the report rather than papering over.