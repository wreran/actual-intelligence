# How papers are ranked, and what the ranking cannot tell you

`find_papers.py` produces three numbers per paper. They mean different
things and are deliberately kept apart.

```
final_score = impact_score x (0.25 + 0.75 x relevance)
```

- **`impact_score`** — how much the research community rates this work.
- **`relevance`** — how much it can plausibly help *your* task.
- **`final_score`** — the ranking key.

## Why multiplicative

An early additive version of this scorer ranked *"Comparative analysis of
CatBoost, LightGBM, XGBoost, RF and DT optimised with PSO to estimate the
number of k-barriers for intrusion detection in wireless sensor networks"*
**third**, on 65 citations and zero topical overlap. That paper is real,
well cited, and worth nothing to a text-classification project.

Additive scoring lets a high citation count buy its way past irrelevance.
Multiplication cannot: relevance near zero drives the product near zero
regardless of citations. The `0.25` floor stops the reverse failure — a
keyword-rich but trivial paper outranking a landmark.

---

## The impact half

| Component | Weight | Source | Why |
|---|---|---|---|
| `velocity` | 0.30 | citations ÷ months since publication | The only citation measure that is fair across an 18-month window |
| `citations` | 0.24 | raw count, log-damped | Rewards a genuine early hit |
| `venue` | 0.30 | venue name → tier table | The key *non-citation* signal; the only real evidence for a 2-month-old paper |
| `access` | 0.10 | OA status + PDF availability | You cannot learn from what you cannot read |
| `influential` | 0.14* | Semantic Scholar `influentialCitationCount` | Filters perfunctory citations from ones that built on the work |
| `percentile` | 0.14* | OpenAlex `cited_by_percentile_year` | Field- and year-normalised standing |

\* Present only when the API returned a value; weights renormalise.

### The recency trap

**Citation counts cannot rank recent papers, and this is not fixable.**

A paper published 3 months ago has near-zero citations whether it is
brilliant or worthless — citation accrual lags publication by 12–24 months.
Ranking an 18-month window by citations therefore ranks mostly by *age*.

Three mitigations, none complete:

1. **Velocity over raw count** — normalises by months available.
2. **Age cohorts** — `SHORTLIST.md` groups into 0–6 / 6–12 / 12–18 months.
   *Compare within a cohort, never across one.* The 0–6 group is explicitly
   labelled "too new to cite — judge on venue/novelty."
3. **Age-neutral signals** — venue and access together carry 0.40, so a
   recent NeurIPS paper can outrank an older uncited one.

### Fields that sound useful but are not

- **`fwci`** (field-weighted citation impact) — OpenAlex's own recency- and
  field-normalised metric, and exactly what this problem calls for. It is
  `null` for most works under ~12 months old. Checked and not relied upon.
- **`cited_by_percentile_year`** — a `{min, max}` band. Taking `.max` pins
  most cited papers at 1.0 and destroys the signal; the midpoint is used.
- **h-index / journal impact factor** — author- and venue-level proxies for
  a paper-level question. A weak paper in a strong journal still scores
  high. Not used.
- **Altmetric** — measures social-media attention, which for ML papers
  tracks controversy and press releases more than quality. Not used.

---

## The relevance half

Two-tier, and the tiering is what makes it work.

**CORE terms** mark a paper as either about text/AI-text detection, or
about boosting *methodology* that transfers. **At least one CORE hit is
required** — a paper with none scores exactly 0.0 and is gated out.

**BONUS terms** are the generic vocabulary of applied ML — "ensemble",
"cross-validation", "hyperparameter". Every paper that ever ran a model
contains these. Capped at 3.0 points so they refine a ranking but never
drive it. Without the cap, an air-quality forecasting paper scored 0.80
relevance on generic vocabulary alone.

**Application-domain penalty (×0.35).** If the *title* names a strong
application domain (medical, wildfire, intrusion detection, credit
scoring, ...) the score is cut hard. "We applied XGBoost to X" studies are
numerous, often well cited, and methodologically empty for your purposes.
The penalty is a multiplier rather than a filter because a paper can be
both domain-specific *and* methodologically interesting.

**Title bonus (+0.15).** Boosting named in the title means it is the
subject of the paper, not one row in a baseline table.

Tune with `--min-topic` (default 4.0). Raise to tighten, lower to widen.
The exact term lists are at the top of `find_papers.py` and are meant to be
edited.

---

## Course-compliance flags

The 50.007 brief forbids deep learning, including LLMs, for Task 3. Each
paper is flagged:

| Flag | Meaning | Action |
|---|---|---|
| `OK` | No neural machinery detected | Usable as-is |
| `PARTIAL` | Neural terms present alongside boosting | The boosting half usually transfers — read it |
| `BLOCKED` | Headline contribution looks neural | Excluded by default (`--include-blocked` overrides) |

**These are keyword heuristics on title and abstract.** They will produce
both false positives and false negatives. A paper that feeds BERT
embeddings into XGBoost is flagged `PARTIAL` and is genuinely half-usable:
take the boosting and calibration, leave the embeddings. Confirm by
reading before relying on any flag.

---

## What this ranking is not

It is a **triage tool**. It reduces ~1000 hits to ~50 worth a look, and
orders them plausibly. It does not read the papers, cannot judge whether a
method is sound, and cannot tell whether a result replicates.

It also cannot see:

- Papers with no OpenAlex or arXiv record
- Quality of the experimental design
- Whether the reported gain survives a fair baseline
- Retracted-in-practice work not yet flagged as retracted
- Whether released code actually runs

The verification pass in `SKILL.md` step 3 exists because of these limits.
The repo already contains a worked example of the discipline —
`Verification Report  XGBoost AI-Text Detection Research Claims.md`, which
caught a venue misattribution, an unreviewed preprint being cited as
peer-reviewed, and a fabricated architectural detail. Apply the same
scepticism here.
