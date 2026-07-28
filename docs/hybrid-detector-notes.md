# Hybrid AI-Text Detector

A single pipeline combining the verified best components of the three curated papers, tuned for the course's Macro-F1 leaderboard task (binary: 0 = human, 1 = AI).

## What was taken from each paper

| Source | Component taken | Where |
|---|---|---|
| SKDU at De-Factify 4.0 (AAAI-25, [arXiv:2503.22338](https://arxiv.org/abs/2503.22338)) | Length-normalized stylometric features (NELA lineage) + XGBoost as the classifier of choice. Their strongest result (F1 0.9979 dev / 0.9945 test) used exactly this pairing. | `features_stylometric.py` |
| NELA toolkit (WWW 2018) | Feature-group design: style, complexity/readability, function-word profile, all normalized by length. Optional hook to the real `nela_features` package. | `features_stylometric.py` |
| LIWC (proprietary) | Replaced with open LIWC-like category rates (pronouns, articles, negations, hedges, certainty). Short-text caveat handled by emitting `n_tokens` so the booster can discount tiny documents. | `features_stylometric.py` |
| DivEye (TMLR 2026, [arXiv:2509.18880](https://arxiv.org/abs/2509.18880)) | The **corrected** 9-dim surprisal-diversity vector: mean/var/skew/kurtosis of surprisal; mean/var of 1st-order diffs; **var/entropy/autocorrelation** of 2nd-order diffs (not repeated mean/var — this was the fix from the verification pass). GPT-2-small backbone by default. | `features_surprisal.py` |
| Fast-DetectGPT (ICLR 2024, [arXiv:2310.05130](https://arxiv.org/abs/2310.05130)) | Analytic conditional probability curvature (the sampling-free estimator that makes it 340x faster than DetectGPT) as one extra feature. | `features_surprisal.py` |
| NotAI.AI ([arXiv:2603.05617](https://arxiv.org/abs/2603.05617), unreviewed preprint) | The meta-architecture: interpretable features → XGBoost → SHAP attribution (`--with-shap`). | `train_hybrid.py` |
| RAID (ACL 2024) | Imbalance insurance: `scale_pos_weight = n_neg / n_pos` always set from the observed class ratio. | `train_hybrid.py` |
| MAGE (ACL 2024) | Decision-threshold calibration on a held-out slice, maximizing Macro-F1 instead of defaulting to 0.5 — cheap OOD recovery. | `train_hybrid.py` |
| Kaggle/community consensus (practitioner folk wisdom — cite as such, not as a paper) | Optuna TPE search space grouped by tree structure → sampling → regularization → learning-rate/trees; StratifiedKFold CV scored on **Macro-F1**, the actual competition metric. | `train_hybrid.py` |

## The math (quick reference)

Surprisal of token \(t\) under a frozen LM \(p\):
\(s_t = -\log_2 p(x_t \mid x_{<t})\), with \(d^{(1)}_t = s_{t+1}-s_t\) and \(d^{(2)}_t = d^{(1)}_{t+1}-d^{(1)}_t\).

DivEye vector: \([\,\mu(s), \sigma^2(s), \text{skew}(s), \text{kurt}(s), \mu(d^{(1)}), \sigma^2(d^{(1)}), \sigma^2(d^{(2)}), H(d^{(2)}), \rho_1(d^{(2)})\,]\), where \(H\) is histogram Shannon entropy and \(\rho_1\) is lag-1 autocorrelation.

Fast-DetectGPT conditional probability curvature (analytic, same-model scoring):

\[
\mathrm{CPC}(x) = \frac{\sum_t \log p(x_t\mid x_{<t}) - \sum_t \mu_t}{\sqrt{\sum_t \sigma_t^2}},
\quad
\mu_t = \sum_{v \in V} p(v\mid x_{<t}) \log p(v\mid x_{<t}),
\quad
\sigma_t^2 = \sum_{v} p(v)\log^2 p(v) - \mu_t^2 .
\]

Machine text scores CPC ≫ 0; human text near or below 0 (verified exactly in `test_smoke.py` against a hand-computed 3-word-vocabulary case).

## Usage

```bash
pip install numpy scipy pandas scikit-learn xgboost optuna
# optional extras:
pip install torch transformers        # for --with-lm (DivEye + CPC features)
pip install shap                      # for --with-shap
pip install nela_features             # for --use-nela-pkg (original NELA vector)

# Course-safe run (pure statistical features, no neural anything):
python train_hybrid.py --train train.csv --test test.csv --trials 60

# Full hybrid (frozen GPT-2 feature extraction — get TA sign-off first):
python train_hybrid.py --train train.csv --test test.csv \
    --with-lm --lm-name gpt2 --with-shap --trials 60
```

Outputs land in `runs/hybrid/`: `model.json`, `run_meta.json` (best params, CV Macro-F1, calibrated threshold), `predictions.csv`, and `shap_top30.txt`.

## Compliance ladder (which flags to use)

1. **Strictest reading of "no deep learning"** — default flags. Stylometric-only; SKDU showed this alone hits F1 ≈ 0.99+ on a near-identical task. Zero risk.
2. **Frozen LM allowed for feature extraction** — add `--with-lm`. This is what DivEye and NotAI.AI do; the *classifier* is still XGBoost. Gray area: confirm with your TA.
3. Either way, run `--with-shap` for the report — SHAP importance gives you exactly the "explain your model" material the Task 4 rubric rewards.

## Files

- `features_stylometric.py` — SKDU/NELA-lineage extractor (39 features, pure numpy)
- `features_surprisal.py` — DivEye [9] + Fast-DetectGPT CPC [2] (pure-math core separated from torch plumbing, so it's unit-testable without a GPU)
- `train_hybrid.py` — feature assembly (with caching), Optuna TPE + stratified Macro-F1 CV, threshold calibration, SHAP, prediction export
- `test_smoke.py` — passing smoke tests for all pure-math components
