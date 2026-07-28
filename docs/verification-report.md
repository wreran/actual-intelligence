# Verification Report: XGBoost-Based AI-Generated Text Detection Research Claims

This report cross-checks five claims (and 8 embedded lineage sub-claims) from a prior research session against primary sources — arXiv preprints, ACL Anthology / AAAI records, and the original benchmark papers. Each item receives a verdict of **VERIFIED**, **PARTIALLY CORRECT**, or **NOT FOUND / LIKELY HALLUCINATED**.

## Summary Table

| # | Claim | Verdict |
|---|---|---|
| 1 | SKDU De-Factify study, XGBoost+NELA, F1≈0.998 | **VERIFIED** (minor rounding note) |
| 2 | DivEye, TMLR 2026, 9-feature surprisal vector, XGBoost meta-classifier | **PARTIALLY CORRECT** |
| 3 | NOTAI.AI paper/system | **VERIFIED** (exists, but overstates its status) |
| 4a | NELA Toolkit — 2018 Web Conference paper | **VERIFIED** |
| 4b | RAIDAR — ICLR 2024, rewriting distance | **VERIFIED** |
| 4c | LIWC — short-text reliability caveats | **VERIFIED** |
| 4d | DetectGPT — 2023, probability curvature (Mitchell et al.) | **VERIFIED** |
| 4e | Fast-DetectGPT — ~340x speedup, conditional probability curvature | **VERIFIED** |
| 4f | GPT-2 as backbone LM for surprisal features | **NOT FOUND / LIKELY HALLUCINATED (as stated)** |
| 4g | RAID benchmark — 2.86%/97.14% split | **VERIFIED** |
| 4h | MAGE/HC3 + "perplexity is leaky" finding | **PARTIALLY CORRECT** |
| 4i | ModernBERT — replication questioning DeBERTaV3 comparison | **VERIFIED** |
| 5 | Kaggle/Optuna XGBoost tuning consensus | **PARTIALLY CORRECT** (real folk wisdom, not a single citable source) |

---

## Claim 1: "SKDU De-Factify" Study

**Verdict: VERIFIED**, with one small precision correction.

The paper is real: **"SKDU at De-Factify 4.0: Natural Language Features for AI-Generated Text-Detection"** by Shrikant Malviya, Pablo Arnau-González, Miguel Arevalillo-Herráez, and Stamos Katsigiannis (Durham University), posted to arXiv on 28 March 2025 ([arXiv:2503.22338](https://arxiv.org/abs/2503.22338v1)). It was presented at the **De-Factify 4.0 workshop**, co-located with the **39th AAAI Conference on Artificial Intelligence (AAAI-25)**, held Feb 25–Mar 4, 2025 in Philadelphia, PA — confirming the "De-Factify workshop, likely AAAI" attribution exactly ([AAAI-25 Workshop Program](https://aaai.org/conference/aaai/aaai-25/workshop-program/); [Durham University repository record](https://durham-repository.worktribe.com/output/3488998/skdu-at-de-factify-40-natural-language-features-for-ai-generated-text-detection)).

The method matches the claim closely: a pipeline combining RAIDAR-inspired rewriting features and **NELA-toolkit-derived content features** — including readability, punctuation patterns, average word length, stopwords, and **psychological features via the LIWC dictionaries** — feeding into classifiers including **XGBoost** ([arXiv:2503.22338](https://ar5iv.labs.arxiv.org/html/2503.22338)).

On the F1 metric, the paper's exact reported numbers for the binary task (Task A) are:
- **0.9979** F1 on the development/challenge leaderboard
- **0.9945** F1 on the final testing leaderboard

Both round to "≈0.998" but the claimed figure of "F1 ≈ 0.998" is closer to the development-set score (0.9979) than the test-set score (0.9945); the prior researcher's number is a reasonable but slightly imprecise summary rather than an exact quote. The paper explicitly states NELA features "significantly outperformed RAIDAR-based features across both tasks," directly matching the "outperforming rewriting-based features" claim ([arXiv:2503.22338](https://ar5iv.labs.arxiv.org/html/2503.22338)).

---

## Claim 2: "DivEye" Paper (TMLR 2026)

**Verdict: PARTIALLY CORRECT.**

The paper is real: **"Diversity Boosts AI-Generated Text Detection"** (system name: DivEye) by Advik Raj Basani and Pin-Yu Chen, first posted 23 Sep 2025, with a v3 revision dated 25 Feb 2026 explicitly noting **"Accepted to Transactions on Machine Learning Research (TMLR '26)"** ([arXiv:2509.18880](https://arxiv.org/abs/2509.18880)). The TMLR 2026 venue claim is therefore accurate. A related, earlier shared-task paper — **"DivEye at PAN 2025: Diversity Boosts AI-Generated Text Detection"** — was separately presented at the Generative AI Authorship Verification Task at **PAN 2025 / CLEF 2025** ([IBM Research](https://research.ibm.com/publications/diveye-at-pan-2025-diversity-boosts-ai-generated-text-detection)); this is a distinct, earlier version of the work.

The feature-vector description is **partially inaccurate**. The paper does define a **9-dimensional feature vector**, but it is not simply "mean/variance/skewness/kurtosis... plus first/second-order temporal differences" as a symmetric extension. The actual feature set is:
- **Distribution (4):** mean, variance, skewness, kurtosis of surprisal
- **1st-order (2):** mean and variance of first-order differences of surprisal
- **2nd-order (3):** variance, **entropy**, and **autocorrelation** of second-order differences of surprisal

So the second-order group includes entropy and autocorrelation terms, not a simple repetition of mean/variance — the claim's description oversimplifies and slightly mischaracterizes the exact composition ([arXiv:2509.18880 full text](https://arxiv.org/html/2509.18880v1)).

The XGBoost meta-classifier claim is confirmed: the paper explicitly trains "a lightweight XGBoost classifier as a meta-model," both standalone (using only DivEye features) and in a boosted configuration where DivEye features are concatenated with existing detectors' (RADAR, DetectLLM, Fast-DetectGPT, Binoculars, BiScope) prediction scores ([arXiv:2509.18880 full text](https://arxiv.org/html/2509.18880v1)).

---

## Claim 3: "NOTAI.AI" Paper/System

**Verdict: VERIFIED** (paper exists), with an important caveat about its status.

**"NotAI.AI: Explainable Detection of Machine-Generated Text via Curvature and Feature Attribution"** is a real preprint by Oleksandr Marchenko Breneur, Adelaide Danilov, Aria Nourbakhsh, and Salima Lamsiyah, submitted to arXiv on 5 March 2026 ([arXiv:2603.05617](https://arxiv.org/html/2603.05617v1)). The system architecture matches closely: it extends **Fast-DetectGPT**, extracting **17 interpretable features** (including **Conditional Probability Curvature**, a **ModernBERT** detector score, readability metrics, and stylometric cues) into an **XGBoost meta-classifier**, with **SHAP**-based explanations and an LLM-based natural-language rationale layer.

Caveat: the paper does not state any accepted publication venue (no conference/workshop/journal listed) — it is currently an **unreviewed arXiv preprint only**, so treating it as an established "paper" without noting this is a peer-review status is a minor overstatement. Note also there is significant name confusion in this space: separate, unrelated commercial products called "NotAI" (a bot-detection pixel/text-monitor service at [isnotai.com](https://www.isnotai.com/how-it-works)) and "ItsNotAI" (an AI-image detector on Hugging Face) exist and should not be conflated with this academic system.

---

## Claim 4: Lineage / Depth-2/3 Claims

### 4a. NELA Toolkit — VERIFIED
The NELA (News Landscape) toolkit paper, **"Assessing the News Landscape: A Multi-Module Toolkit for Evaluating the Credibility of News,"** by Benjamin D. Horne et al., was published at **WWW 2018 (The Web Conference)**, appearing in the ACM digital library under WWW '18 companion proceedings, and its purpose is explicitly news credibility assessment ([ACM DL record](http://dl.acm.org/citation.cfm?doid=3184558.3186987); [author's PDF](https://benjamindhorne.github.io/pdfs/WWW18_Horne_Demo.pdf)). This matches the claim exactly.

### 4b. RAIDAR — VERIFIED
**"Raidar: geneRative AI Detection viA Rewriting"** by Chengzhi Mao, Carl Vondrick, Hao Wang, and Junfeng Yang was accepted at **ICLR 2024** ([arXiv:2401.12970](https://arxiv.org/abs/2401.12970)). The core mechanism matches precisely: "large language models (LLMs) are more likely to modify human-written text than AI-generated text when tasked with rewriting," because LLMs perceive AI-generated text as already high quality — i.e., LLMs rewrite AI text less than human text, exactly as claimed.

### 4c. LIWC — VERIFIED
LIWC (Linguistic Inquiry and Word Count) is confirmed as a proprietary psycholinguistic dictionary tool with documented short-text and general reliability limitations. Independent evaluation studies report precision falling "as low as 49.6% and recall as low as 41.7% for some categories" ([Hunter & Grant, Aston University](https://research.aston.ac.uk/files/199481203/Hunter_and_Grant_is_LIWC_reliable_efficient_and_effective_for_the_analysis_of_large_online_datasets_question.pdf)), and the tool's own documentation notes it treats very short texts as unreliable ("minimal data does not contain enough content ... LIWC-22 considers this data as missing rather than providing an inaccurate score") ([liwc.app](https://www.liwc.app/help/liwc)).

### 4d. DetectGPT — VERIFIED
**"DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"** by Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and Chelsea Finn was published in **2023**, at **ICML 2023** ([arXiv:2301.11305](https://arxiv.org/abs/2301.11305)), matching the claim exactly on year, authorship, and method (probability-curvature-based detection via perturbations).

### 4e. Fast-DetectGPT — VERIFIED
**"Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature"** by Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang was accepted at **ICLR 2024** ([arXiv:2310.05130](https://arxiv.org/abs/2310.05130)). The abstract explicitly states it "accelerates the detection process by a factor of 340," confirming the ~340x speedup figure, and it introduces "conditional probability curvature" as its core innovation — both details match the claim exactly.

### 4f. GPT-2 as Backbone LM for Surprisal Features — NOT FOUND AS STATED
No direct evidence was found tying GPT-2 specifically to the surprisal features described in DivEye or NotAI.AI. The NotAI.AI demo video instead references **"a GPT-2-1.3B token probability distribution"** as a proxy model for its curvature calculation ([YouTube demo transcript](https://www.youtube.com/watch?v=9ZPaYtJlOXU)) — but "GPT-2 1.3B" is not a standard, publicly released GPT-2 size (standard GPT-2 variants are 124M/355M/774M/1.5B parameters); this appears to be either a transcription/demo inconsistency or an internal naming choice not otherwise documented in the paper text. The claim that GPT-2 is used generically "as backbone LM for surprisal features" across this research lineage is too broad and not clearly substantiated as a general pattern — treat this sub-claim with caution.

### 4g. RAID Benchmark — 2.86% / 97.14% Split — VERIFIED
The RAID paper, **"RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors"** by Liam Dugan et al. ([arXiv:2405.07940](https://arxiv.org/abs/2405.07940); [ACL Anthology PDF](https://aclanthology.org/2024.acl-long.674.pdf)), reports **14,971 human-written documents** versus **509,014 AI-generated documents** in its non-adversarial base dataset. Computing the exact percentages:

\[
\frac{14{,}971}{523{,}985} \times 100 \approx 2.857\%,\qquad \frac{509{,}014}{523{,}985}\times 100 \approx 97.143\%
\]

This confirms the claimed **2.86% / 97.14%** split almost exactly (a commonly cited shorthand for this is "roughly a 40:1 ratio" of AI-generated to human-written text). Note this ratio describes the base (non-adversarial) dataset; once adversarial attacks are included the total balloons to over 6 million generations, though the underlying human-document count and imbalance ratio persist.

### 4h. MAGE, HC3, and "Perplexity Is Leaky" — PARTIALLY CORRECT
Both benchmarks are real: **MAGE** ("Machine-generated Text Detection in the Wild," Yafu Li et al., ACL 2024, [aclanthology.org/2024.acl-long.3](https://aclanthology.org/2024.acl-long.3)) and **HC3** ("How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection," Guo et al., 2023, [Semantic Scholar](https://www.semanticscholar.org/paper/cb29cf52f0f7d2e4324c68690a55b22890f2212d)).

However, the characterization of the finding is imprecise. MAGE's paper does **not** simply conclude perplexity is a "leaky"/unreliable signal in a blanket sense — it actually finds perplexity is a **useful clustering feature** for distinguishing human vs. machine text ("perplexity can serve as a fundamental feature for clustering the two sources of text... applicable ... regardless of the text domain or the language model used for generation"), while separately noting a caveat that **"perplexity bias can hinder robust detection"** and that PLM-based detectors "exhibit overconfidence in text perplexity" ([arXiv:2305.13242](https://arxiv.org/html/2305.13242v3)). The "leaky signal" framing is a reasonable colloquial gloss on this bias/overconfidence finding, but attributing a flat "perplexity alone is unreliable" conclusion directly to MAGE overstates and slightly distorts what the paper actually argues; it is closer to a synthesis of MAGE's caveats plus broader community findings from other work (e.g., studies on data leakage/teacher-forcing effects in perplexity-based detection) than a single explicit MAGE conclusion.

### 4i. ModernBERT vs. DeBERTaV3 Replication — VERIFIED
**"ModernBERT or DeBERTaV3? Examining Architecture and Data Influence on Transformer Encoder Models Performance"** by Wissam Antoun, Benoît Sagot, and Djamé Seddah ([arXiv:2504.08716](https://arxiv.org/abs/2504.08716)) is exactly the independent replication described. Its controlled study — pretraining ModernBERT on the same data as CamemBERTaV2 (a DeBERTaV3 French model) — finds that "the previous model generation remains superior in sample efficiency and overall benchmark performance," directly questioning ModernBERT's claimed advantages, while confirming ModernBERT's genuine edges are long-context support and faster training/inference speed.

---

## Claim 5: Kaggle/Community XGBoost Tuning Consensus

**Verdict: PARTIALLY CORRECT — directionally accurate community practice, but not tied to any single verifiable canonical source.**

The specific tactical claims are individually well-supported by widely circulated XGBoost tuning guidance:

- **Tune tree-depth/regularization before learning rate**: This staged approach (fix a moderate learning rate, tune `max_depth`/`min_child_weight`, then regularization, then lower the learning rate at the end) is standard, widely repeated guidance (e.g., [MetricGate's XGBoost tuning guide](https://metricgate.com/blogs/xgboost-hyperparameter-tuning-guide/)).
- **Optuna preferred over grid search**: Broadly true and heavily documented — Optuna's Tree-structured Parzen Estimator (TPE) sampler and pruning let it find comparable or better results with far fewer trials than exhaustive grid search ([Druce.ai XGBoost/Optuna benchmark](https://druce.ai/2020/10/hyperparameter-tuning-with-xgboost-ray-tune-hyperopt-and-optuna); [Optuna documentation](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)).
- **~5–15x speedup claims**: This specific range is a plausible summary of commonly cited figures, but there is no single canonical benchmark stating exactly "5-15x." Reported figures vary widely by source and context — one hands-on comparison found Optuna running roughly 2x as many trials in about half the time versus sequential/grid search ([Druce.ai](https://druce.ai/2020/10/hyperparameter-tuning-with-xgboost-ray-tune-hyperopt-and-optuna)), a 2026 tutorial states Optuna "typically finds better hyperparameters in 10–20% of the trials grid search would require" (i.e., a 5–10x reduction in trials) ([GUVI Optuna vs Grid Search guide](https://www.guvi.in/blog/optuna-for-hyperparameter-optimization/)), and Optuna's own internal sampler updates report speedups as high as 300x for a specific multi-objective TPE implementation detail unrelated to XGBoost tuning generally ([Optuna Medium post on TPESampler v4.0.0](https://medium.com/optuna/significant-speed-up-of-multi-objective-tpesampler-in-optuna-v4-0-0-2bacdcd1d99b)). The "5-15x" figure in the original claim is a reasonable midpoint estimate of scattered community reports, not a quote from any specific paper or Kaggle post.
- **Align CV and `scale_pos_weight` with Macro-F1 metric**: This is standard, correct imbalanced-classification practice (using class weighting via `scale_pos_weight` and evaluating with Macro-F1 rather than accuracy on imbalanced data is widely recommended across academic and applied XGBoost tuning literature, e.g. [Imbalance-XGBoost](https://arxiv.org/abs/1908.01672), [Evaluating XGBoost for Balanced and Imbalanced Data](https://arxiv.org/abs/2303.15218)), but again reflects general best practice rather than a single identifiable source.

**Overall for Claim 5**: the substance is accurate as a description of community consensus, but the original research session should not have presented this as if backed by one traceable citation — it should be flagged as synthesized "folk wisdom" rather than a specific paper/tool claim (unlike Claims 1–4, which map to identifiable papers).

---

## Key Corrections and Flags for the Prior Research Session

1. **Claim 1 (SKDU)**: Essentially accurate; F1≈0.998 is a slight rounding of the actual 0.9979 (dev) / 0.9945 (test) figures — should be cited more precisely.
2. **Claim 2 (DivEye)**: TMLR 2026 venue is correctly identified, but the 9-feature description ("mean/variance/skewness/kurtosis... plus first/second-order temporal differences") oversimplifies the actual feature set, which includes entropy and autocorrelation terms in the second-order group, not merely repeated mean/variance statistics.
3. **Claim 3 (NOTAI.AI)**: Exists essentially as described, but is an unreviewed arXiv preprint with no stated acceptance venue — should not be treated as a peer-reviewed publication. Also watch for confusion with unrelated commercial "NotAI" bot-detection and "ItsNotAI" image-detection products.
4. **Claim 4f (GPT-2 backbone)**: Not clearly substantiated as a general pattern across this research lineage; the only supporting detail found references an unusual "GPT-2-1.3B" proxy model in a demo video transcript, which does not match standard GPT-2 model sizes.
5. **Claim 4h (perplexity "leaky")**: MAGE's actual finding is more nuanced — perplexity is described as a useful clustering signal with an important bias/overconfidence caveat, not a flatly unreliable one. The "leaky" framing should be attributed to broader community synthesis, not to MAGE specifically.
6. **Claim 5 (Optuna speedup)**: The "~5-15x" figure is a reasonable estimate but not traceable to one canonical source; should be labeled as general practitioner consensus rather than a specific citation.

All other elements of the claims — venues, years, authors, and core mechanisms for SKDU, DivEye, NotAI.AI, NELA, RAIDAR, LIWC, DetectGPT, Fast-DetectGPT, RAID, MAGE, HC3, and the ModernBERT/DeBERTaV3 replication — check out as accurate against primary sources.
