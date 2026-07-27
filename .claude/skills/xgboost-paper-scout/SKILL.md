---
name: xgboost-paper-scout
description: Find recent, high-impact XGBoost / gradient-boosting papers within a rolling recency window (default 18 months), rank them by citation velocity, venue and task-relevance, download the open-access PDFs into the project, build a runnable demo that tests each paper's technique against the current baseline, and apply anti-overfitting tuning. Use when asked to find/scout/survey recent ML or XGBoost papers, find impactful or highly-cited recent work, download papers to read, reproduce or demo a paper's technique, or tune a boosted-tree model without overfitting.
---

# XGBoost paper scout

Turns "find me recent XGBoost papers worth learning from" into: a ranked
shortlist, PDFs on disk, a demo that honestly measures whether each
technique helps, and a tuning pass that does not buy leaderboard score with
generalisation.

Five stages. Run them in order; each consumes the previous one's output.

## Prerequisites

Network access is required (OpenAlex, arXiv, Unpaywall). The scripts use
only the Python standard library, so no install is needed to search or
download. Demos additionally need whatever the project already uses
(`xgboost`, `scikit-learn`, `kagglehub`).

If `WebSearch`/`WebFetch` are available, use them in stage 3 — the
verification pass is much weaker without them, and this skill exists partly
to prevent confidently-cited nonsense.

---

## Stage 1 — Search

```bash
python3 .claude/skills/xgboost-paper-scout/scripts/find_papers.py \
    --months 18 --top 20 --out-dir papers --mailto <your-email>
```

Queries OpenAlex and arXiv, de-duplicates, then applies three hard gates:
recency, gradient-boosting family, and topic relevance. Writes
`papers/SHORTLIST.md` (ranked, human-readable, grouped by age cohort) and
`papers/candidates.json` (everything, with score breakdowns).

Useful flags:

| Flag | Use |
|---|---|
| `--months N` | Change the window. Default 18. |
| `--preset generic` | Drop the AI-text-detection bias; method papers only. |
| `--query "..."` | Add a search string. Repeatable. |
| `--min-topic N` | Relevance gate, default 4.0. Raise to tighten. |
| `--include-blocked` | Keep papers flagged as deep-learning-based. |
| `--no-s2` | Skip Semantic Scholar (it rate-limits without a key). |

**Report the funnel to the user** — raw hits, survivors of each gate,
shortlist size. If a gate removed almost everything, say so and widen
rather than silently returning three papers.

## Stage 2 — Rank and read the shortlist

Ranking is `final_score = impact_score x (0.25 + 0.75 x relevance)`.
Read `references/impact-scoring.md` before defending or adjusting any
number. The two things to internalise:

- **Citation counts cannot rank papers under ~12 months old.** Compare
  within an age cohort, never across. The 0–6 month group is judged on
  venue and novelty, and the shortlist labels it that way.
- **The scorer is triage, not judgement.** It narrows ~1000 hits to ~50.
  Choosing among those 50 is stage 3, and it requires reading.

Open `papers/SHORTLIST.md` and read the top 10–15 entries — title, venue,
topic hits, course flag. Discard anything whose relevance is an artefact of
keyword collision.

## Stage 3 — Verify before trusting

Keyword heuristics produce false positives. Before recommending a paper,
confirm by reading — abstract at minimum, method section for anything you
plan to demo:

1. **Venue and peer-review status.** An arXiv preprint is not a
   peer-reviewed paper. Say which it is.
2. **Is the contribution actually about boosting**, or is XGBoost one row
   in a baseline table?
3. **Course compliance.** The brief states *"DO NOT use any deep learning
   approach (including LLMs)"* for Task 3. A `PARTIAL` flag usually means
   the boosting half transfers and the neural feature extractor does not —
   state exactly which part you are taking.
4. **Does the claimed gain have a fair baseline?** A technique that beats
   an untuned XGBoost has demonstrated nothing.

This repo contains a worked example of the standard —
`Verification Report  XGBoost AI-Text Detection Research Claims.md` — which
caught a real venue misattribution, a preprint cited as peer-reviewed, and
a fabricated architectural detail. Match that level of scepticism, and flag
uncertainty rather than smoothing it over.

## Stage 4 — Download

```bash
python3 .claude/skills/xgboost-paper-scout/scripts/fetch_pdf.py \
    --from-json papers/candidates.json --top 5 \
    --out-dir papers --mailto <your-email>
```

PDFs land in `papers/pdf/`, with `papers/MANIFEST.md` recording what
arrived, from where, and what did not.

**Open access only.** The fetcher follows OA links published by OpenAlex,
arXiv or Unpaywall. Publisher 403s are recorded as failures, not worked
around. Do not extend this to proxies, credentials or scrapers — if a paper
is paywalled, point the user at their university library.

Expect roughly 60–80% success. Some publishers block automated agents even
for genuinely open-access articles.

## Stage 5 — Demo, then tune

**Demo.** Copy `assets/demo_template.py` to `demos/demo_<slug>.py` and fill
it in. The template enforces a paired comparison: baseline and the paper's
technique on identical folds, threshold calibration applied to both arms,
and the mean gain reported against the paired standard error.

Change **one mechanism** per demo. Changing the technique and the
hyperparameters together makes the result unattributable.

Report the verdict the template computes — including when it says the gain
is inside the noise floor. A negative result is a real finding and the
Task 4 rubric explicitly rewards the roadmap over the wins.

**Tune.** Then work through `references/overfitting-playbook.md`. It is
ordered by score-per-unit-risk and opens with three concrete defects in
`task3_xgboost_leaderboard.py` — selection bias from reporting
`study.best_value`, no early stopping, and a missing `output/` directory —
each of which silently inflates the reported number.

The highest-value moves, in order: hold out a slice Optuna never sees;
calibrate the decision threshold instead of using 0.5; seed-average the
final prediction; widen `min_child_weight` and lower `colsample_bytree` for
wide TF-IDF input.

---

## Project context

Tuned for the SUTD 50.007 GenAI-text-detection project in this repo.

- **Metric** is Macro-F1, not accuracy. XGBoost optimises logloss, so
  threshold calibration is nearly free score.
- **Task 3 forbids deep learning and LLMs.** This constrains which papers
  are usable and is enforced by default in stage 1.
- **Task 3's top rubric band needs more than 4 models** with documented
  hyperparameter tuning — so breadth is worth marks, not just depth on
  XGBoost. The playbook's §6 lists decorrelated candidates.
- **Data is <5% of the source dataset.** Published numbers on full
  De-Factify data (~0.99 F1) are not a reachable target here, and chasing
  them is the overfitting failure mode the brief warns about.

For a different project, run with `--preset generic` and edit the term
lists at the top of `find_papers.py`.

## Files

| Path | Purpose |
|---|---|
| `scripts/find_papers.py` | Search, gate, score, shortlist (stdlib only) |
| `scripts/fetch_pdf.py` | OA-only PDF downloader + manifest (stdlib only) |
| `references/impact-scoring.md` | What each metric means and cannot support |
| `references/overfitting-playbook.md` | Tuning ordered by score-per-unit-risk |
| `assets/demo_template.py` | Paired baseline-vs-paper demo harness |
