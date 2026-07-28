# 50.007 Machine Learning — Final Report Scaffold
## GenAI Content Detection Project

**Team members:** _____________________
**Date:** _____________________

---

## Table of Contents
1. Introduction & Problem Statement
2. Task 1: Logistic Regression from Scratch
3. Task 2: PCA + KNN
4. Task 3: Leaderboard Model (XGBoost)
5. Discussion Questions (Task 4 requirements)
6. Difficulties & Reflections
7. Self-Learning Beyond the Course
8. References
9. Appendix

---

## 1. Introduction & Problem Statement

> _Fill in: restate the problem — "Given a piece of text, classify if this is human-authored or machine-generated content," dataset source (GenAIDetect Workshop, COLING 2025), and split sizes (20k train / 2k dev / 5k test)._

- [ ] State why GenAI detection matters (1 paragraph)
- [ ] Describe the dataset provenance and known limitation (only <5% of full dataset sampled — expect lower ceiling than published results)
- [ ] State evaluation metric: **Macro-F1**, not accuracy — explain why this matters for class imbalance

---

## 2. Task 1: Logistic Regression from Scratch

### 2.1 Model Explanation
- [ ] Write out the sigmoid function, hypothesis, and cross-entropy loss (see equations below — you may reuse/adapt)

\[
\sigma(z) = \frac{1}{1+e^{-z}}, \qquad
\hat{y} = \sigma(w^\top x + b)
\]

\[
\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right] + \frac{\lambda}{2m}\|w\|_2^2
\]

- [ ] Describe your gradient descent update rule (batch/mini-batch/stochastic — state which you chose and why)
- [ ] State your chosen learning rate, number of iterations, regularization strength, and how you picked them (grid search? manual?)

### 2.2 Implementation Summary
- [ ] Reference your notebook / code file: `task1_logistic_regression.py` [cite: code_file]
- [ ] Report training loss curve (plot loss vs iteration)
- [ ] Report validation Accuracy and Macro-F1

### 2.3 Comparison to sklearn LogisticRegression
- [ ] Train a `sklearn.linear_model.LogisticRegression` on the same features (for comparison ONLY, not for submission)
- [ ] Tabulate: | Metric | Your Implementation | sklearn |
- [ ] Discuss any performance gap and why it might exist (regularization defaults, solver differences, convergence tolerance)

---

## 3. Task 2: PCA + KNN

### 3.1 Model Explanation
- [ ] Briefly explain PCA (variance-maximizing linear projection, eigenvectors of covariance matrix)
- [ ] Briefly explain KNN (majority vote among k=2 nearest neighbors, distance metric used — default Euclidean)

### 3.2 Required Results Table
**MUST report Macro-F1 for n_components = 2000, 1000, 500, 100 with KNN n_neighbors=2:**

| n_components | Macro-F1 (test set) | Explained Variance Ratio | Notes |
|---|---|---|---|
| 2000 | | | |
| 1000 | | | |
| 500 | | | |
| 100 | | | |

- [ ] Fill in from `task2_pca_knn.py` output (`task2_pca_knn_report_table.csv`)
- [ ] Insert plot: F1 vs n_components (`task2_f1_vs_components.png`)
- [ ] Discuss the trend: does F1 increase or decrease with more components? Why? (bias-variance / curse of dimensionality for KNN in high-dim space)

### 3.3 Analysis
- [ ] Discuss the trade-off between dimensionality reduction and information loss
- [ ] Discuss why KNN is particularly sensitive to dimensionality (distance concentration in high dimensions)

---

## 4. Task 3: Leaderboard Model — XGBoost

### 4.1 Model Explanation — How XGBoost Works
- [ ] Explain gradient boosting: additive ensemble of regression trees, each fit to the negative gradient (residual) of the loss
- [ ] Explain XGBoost's specific contributions: regularized objective, second-order (Newton) approximation, sparsity-aware split finding, weighted quantile sketch [cite: web:17]

\[
\mathcal{L}(\phi) = \sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k), \qquad
\Omega(f) = \gamma T + \tfrac{1}{2}\lambda\|w\|^2
\]

### 4.2 Feature Engineering — What You Tried
- [ ] List every feature family you engineered (use the table from your research report as reference):
  - [ ] TF-IDF (course-provided, required baseline)
  - [ ] Stylometric/complexity features (word count, TTR, punctuation ratios, etc.) — see `extract_stylometric_features()`
  - [ ] (Optional, disclose if used) Surprisal-based DivEye features
  - [ ] Any n-gram extensions, interaction terms, or PCA-reduced features reused from Task 2
- [ ] For each feature family, briefly justify inclusion with a citation to the research (NELA study, DivEye, NOTAI.AI)

### 4.3 Hyperparameter Tuning Roadmap
- [ ] Describe your search strategy (staged tuning vs Optuna — see roadmap below)
- [ ] Table of hyperparameters tried and best found:

| Parameter | Search Range | Best Value | CV Macro-F1 impact |
|---|---|---|---|
| max_depth | 3–12 | | |
| learning_rate | 0.005–0.3 | | |
| n_estimators | 100–600 | | |
| subsample | 0.5–1.0 | | |
| colsample_bytree | 0.5–1.0 | | |
| min_child_weight | 1–10 | | |
| gamma | 0–5 | | |
| scale_pos_weight | class ratio | | |

- [ ] Insert Optuna optimization history plot (trial number vs Macro-F1)
- [ ] Insert final feature importance plot (from `task3_feature_importance.csv`)

### 4.4 Other Models Explored (rubric requires 2-3+ for mid marks, 4+ for full marks)
- [ ] Model 2: _____________ (e.g., Random Forest) — Macro-F1: _____
- [ ] Model 3: _____________ (e.g., SVM / Linear SVC) — Macro-F1: _____
- [ ] Model 4: _____________ (e.g., LightGBM / CatBoost) — Macro-F1: _____
- [ ] Model 5 (optional): _____________ 
- [ ] Comparison table of all models with dev/public leaderboard Macro-F1

### 4.5 Final Leaderboard Result
- [ ] Public leaderboard score: _____
- [ ] Private leaderboard score (if available at report time): _____
- [ ] Screenshot of leaderboard position

---

## 5. Discussion Questions (Required by Task 4)

**Q1: Introduce your best performing model and how it works.**
> _Answer here — reference Section 4.1_

**Q2: How did you achieve your best model? What parameters did you tune, and what roadmap did you follow?**
> _Answer here — reference Section 4.3, describe the staged-tuning-then-Optuna approach and why you chose it_

**Q3: Difficulties faced when tuning models, and how did you overcome (or fail to overcome) them?**
> _Answer here — see Section 6_

**Q4: Did you self-learn anything beyond the course? What and how?**
> _Answer here — see Section 7 (e.g., Optuna Bayesian search, SHAP interpretability, surprisal-based features, gradient boosting math)_

---

## 6. Difficulties & Reflections

- [ ] Describe any compute/time constraints faced (e.g., Optuna trials taking too long, large TF-IDF dimensionality)
- [ ] Describe any data issues (class imbalance, noisy labels, small train set relative to full published dataset)
- [ ] Describe any implementation bugs and how you debugged them (e.g., gradient descent not converging, NaNs in loss)

---

## 7. Self-Learning Beyond the Course

Suggested topics to discuss (pick what you actually explored):
- [ ] Gradient boosting math (second-order Taylor expansion, regularized objective) [cite: web:17]
- [ ] Bayesian hyperparameter optimization via Optuna / TPE sampler [cite: web:9]
- [ ] SHAP values for model interpretability [cite: web from NOTAI.AI paper]
- [ ] Token-level surprisal and information-theoretic text features (DivEye) [cite: web:21]
- [ ] Stylometric/NELA-style linguistic feature extraction [cite: web:20]

---

## 8. References

> _Use consistent citation style (e.g., APA). Include at minimum:_
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
- Malviya, S., et al. (2025). SKDU at De-Factify 4.0: Natural Language Features for AI-Generated Text-Detection.
- Basani, A. R., & Chen, P.-Y. (2026). Diversity Boosts AI-Generated Text Detection.
- Marchenko Breneur, O., et al. (2026). NOTAI.AI: Explainable Detection of Machine-Generated Text via Curvature and Feature Attribution.
- Wang, Y., et al. (2025). GenAI content detection task 1: English and multilingual machine-generated text detection: AI vs. human.

---

## 9. Appendix

- [ ] Full hyperparameter search logs
- [ ] Additional confusion matrices / error analysis
- [ ] Any additional plots
