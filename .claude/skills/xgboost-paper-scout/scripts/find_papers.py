#!/usr/bin/env python3
"""
find_papers.py — recency-bounded, impact-ranked search for gradient-boosting papers.

Standard library only. No pip install required.

Queries OpenAlex (primary: rich metadata, OA PDF links, citation percentiles) and
arXiv (preprint coverage + guaranteed PDF access), merges and de-duplicates the
results, then filters and ranks them.

Three hard gates are applied before scoring:
  1. RECENCY  — publication date inside the rolling window (default 18 months).
  2. FAMILY   — title/abstract must name a gradient-boosted-tree method
                (XGBoost, LightGBM, CatBoost, GBDT, "gradient boosting", ...).
                This is what stops the search collapsing into generic ML.
  3. INTEGRITY— retracted and paratext records are dropped.

Survivors get a transparent composite impact score (see references/impact-scoring.md)
and a course-compliance flag for the "no deep learning" rule.

Usage
-----
  python3 find_papers.py                          # defaults: 18 months, project preset
  python3 find_papers.py --months 12 --top 15
  python3 find_papers.py --preset generic         # drop the AI-text-detection bias
  python3 find_papers.py --query "conformal prediction gradient boosting"
  python3 find_papers.py --out-dir papers --mailto you@example.com

Outputs
-------
  <out-dir>/candidates.json  — every survivor, full metadata + score breakdown
  <out-dir>/SHORTLIST.md     — ranked human-readable table, grouped by age cohort
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Gate 2: the gradient-boosting family. A paper must name one of these.
# ---------------------------------------------------------------------------

GBDT_TERMS = [
    "xgboost", "x-gboost", "lightgbm", "light gbm", "catboost",
    "gradient boosting", "gradient-boosting", "gradient boosted",
    "gradient-boosted", "gbdt", "gbm", "boosted tree", "boosting tree",
    "tree ensemble", "histogram-based boosting",
]

# Topical relevance is two-tier, and the tiering is the whole trick.
#
# CORE terms mark a paper as either (a) about text / AI-text detection, or
# (b) about boosting *methodology* you can lift wholesale. At least one CORE
# hit is required — scored relevance alone does not survive the gate.
#
# BONUS terms are the generic vocabulary of applied ML. Every paper that ever
# ran a model says "hyperparameter", "ensemble", "cross-validation". Left
# ungated they let a paper on air-quality forecasting outrank a paper on
# calibrating boosted trees, which is exactly the failure this split prevents.
PRESET_TOPICS = {
    # Tuned for the 50.007 GenAI-text-detection project.
    "project": {
        "core": [
            ("machine-generated text", 4.0), ("ai-generated text", 4.0),
            ("machine generated text", 4.0), ("llm-generated", 3.5),
            ("text detection", 3.0), ("authorship", 3.0), ("stylometr", 3.5),
            ("text classification", 3.0), ("document classification", 2.5),
            ("natural language", 2.0), ("nlp", 2.0), ("corpus", 1.5),
            ("tf-idf", 3.0), ("n-gram", 2.0), ("bag of words", 2.5),
            # methodology that transfers regardless of domain
            ("macro-f1", 3.0), ("macro f1", 3.0), ("imbalanc", 2.5),
            ("class imbalance", 3.0), ("calibrat", 2.5), ("overfit", 2.5),
            ("regulariz", 2.0), ("hyperparameter optimization", 2.5),
            ("hyperparameter tuning", 2.5), ("optuna", 3.0),
            ("bayesian optimization", 2.5), ("tabular", 3.0),
            ("threshold optimization", 3.0), ("decision threshold", 3.0),
        ],
        "bonus": [
            ("ensemble", 1.0), ("stacking", 1.5), ("shap", 1.0),
            ("feature selection", 1.0), ("interpretab", 0.5),
            ("explainab", 0.5), ("cross-validation", 1.0),
            ("generaliz", 0.5), ("feature engineering", 1.0),
            ("grid search", 0.5), ("class weight", 1.0),
        ],
    },
    # Method-only: no text-domain bias, still excludes domain applications.
    "generic": {
        "core": [
            ("hyperparameter optimization", 3.0), ("hyperparameter tuning", 3.0),
            ("regulariz", 2.5), ("overfit", 3.0), ("calibrat", 3.0),
            ("tabular", 3.0), ("imbalanc", 2.5), ("class imbalance", 3.0),
            ("convergence", 2.0), ("generalization bound", 3.0),
            ("benchmark", 2.0), ("scalab", 2.0), ("split finding", 3.0),
            ("loss function", 2.0), ("boosting theory", 3.5),
        ],
        "bonus": [
            ("ensemble", 1.0), ("feature selection", 1.0), ("shap", 1.0),
            ("interpretab", 0.5), ("cross-validation", 1.0),
        ],
    },
}

# Strong application-domain markers. A paper whose *title* sits in one of these
# and shows no text/NLP core term is an "we applied XGBoost to X" study: real
# work, frequently well cited, and of no methodological use here.
APPLICATION_DOMAINS = [
    "air quality", "sepsis", "diabetes", "cancer", "tumor", "tumour", "clinical",
    "patient", "disease", "medical", "health", "covid", "mortality", "diagnosis",
    "intrusion detection", "wireless sensor", "network traffic", "iot",
    "fault detection", "fault diagnosis", "bearing", "manufacturing",
    "crop", "soil", "agricultur", "yield prediction", "rainfall", "flood",
    "landslide", "groundwater", "seismic", "earthquake", "wind speed",
    "solar", "photovoltaic", "energy consumption", "power load", "battery",
    "concrete", "compressive strength", "corrosion", "material",
    "stock price", "credit scoring", "bankruptcy", "insurance claim",
    "traffic flow", "vehicle", "driver", "aviation", "railway",
    "protein", "gene expression", "molecul", "drug", "chemical", "toxicity",
    "student performance", "churn", "real estate", "housing price",
    "wildfire", "forest fire", "weather", "climate", "pollution", "emission",
    "phishing", "malware", "botnet", "spam filter", "ddos",
    "sports", "tourism", "supply chain", "logistics", "retail sales",
]

# Default search strings sent to the APIs, per preset.
PRESET_QUERIES = {
    "project": [
        "XGBoost text classification",
        "gradient boosting machine-generated text detection",
        "XGBoost imbalanced classification macro F1",
        "gradient boosting hyperparameter optimization overfitting",
        "XGBoost calibration tabular",
        "LightGBM CatBoost benchmark tabular",
        "gradient boosted trees feature selection text",
    ],
    "generic": [
        "XGBoost",
        "gradient boosting decision trees",
        "LightGBM CatBoost comparison",
        "gradient boosting regularization overfitting",
        "gradient boosting hyperparameter optimization",
    ],
}

# ---------------------------------------------------------------------------
# Course-compliance: the 50.007 brief forbids deep learning / LLMs as the model.
# ---------------------------------------------------------------------------

NEURAL_TERMS = [
    "deep learning", "neural network", "transformer", "bert", "roberta",
    "deberta", "modernbert", "large language model", "llm", "gpt",
    "convolutional", "recurrent", "lstm", "gru", "attention mechanism",
    "fine-tun", "finetun", "pretrained language model", "encoder-decoder",
    "self-supervised", "foundation model", "embedding model",
]

# ---------------------------------------------------------------------------
# Venue tiers. Used as a non-citation quality signal — essential for papers too
# recent to have accumulated citations.
# ---------------------------------------------------------------------------

VENUE_TIERS = [
    (1.00, ["neurips", "neural information processing", "icml",
            "international conference on machine learning", "iclr",
            "learning representations", "jmlr", "journal of machine learning research",
            "tmlr", "transactions on machine learning research",
            "kdd", "knowledge discovery and data mining",
            "acl", "annual meeting of the association for computational linguistics",
            "emnlp", "empirical methods in natural language processing",
            "naacl", "aaai", "ijcai", "tpami", "pattern analysis and machine intelligence"]),
    # NB: no bare "machine learning" here — it substring-matches dozens of
    # mid-tier journals ("International Journal of Machine Learning and
    # Cybernetics") and inflates them to near-top-tier.
    (0.80, ["aistats", "artificial intelligence and statistics", "uai",
            "uncertainty in artificial intelligence", "colt", "learning theory",
            "ecml", "pkdd", "cikm", "information and knowledge management",
            "wsdm", "web search and data mining", "sigir", "the web conference",
            "coling", "tacl", "transactions of the association for computational",
            "machine learning journal", "data mining and knowledge discovery",
            "journal of machine learning"]),
    (0.60, ["pakdd", "icdm", "sdm", "ecai", "lrec", "eacl", "aacl",
            "expert systems with applications", "knowledge-based systems",
            "information sciences", "neurocomputing", "ieee access",
            "pattern recognition", "information fusion", "applied soft computing"]),
    (0.40, ["workshop", "symposium", "companion", "shared task", "semeval"]),
]

STOP_TITLE_WORDS = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# HTTP with polite backoff. Semantic Scholar rate-limits aggressively without a
# key; every network call degrades to "skip this source" rather than crashing.
# ---------------------------------------------------------------------------

def http_get(url: str, mailto: str | None = None, retries: int = 3,
             timeout: int = 60) -> bytes | None:
    ua = f"xgboost-paper-scout/1.0 (+https://openalex.org; mailto:{mailto or 'anonymous'})"
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "*/*"})
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    [{e.code}] backing off {wait}s ...", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"    [skip] HTTP {e.code} for {url[:90]}", file=sys.stderr)
            return None
        except Exception as e:  # noqa: BLE001 - network flakiness must not kill a run
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            print(f"    [skip] {type(e).__name__} for {url[:90]}", file=sys.stderr)
            return None
    return None


# ---------------------------------------------------------------------------
# Source 1: OpenAlex
# ---------------------------------------------------------------------------

def reconstruct_abstract(inv: dict | None) -> str:
    """OpenAlex ships abstracts as an inverted index; rebuild linear text."""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def venue_of(work: dict) -> str:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name") or ""


def search_openalex(query: str, since: str, until: str, mailto: str,
                    per_page: int = 100, max_pages: int = 2) -> list[dict]:
    out: list[dict] = []
    for page in range(1, max_pages + 1):
        params = {
            "filter": (f"from_publication_date:{since},to_publication_date:{until},"
                       f"default.search:{query}"),
            "per-page": str(per_page),
            "page": str(page),
            "sort": "cited_by_count:desc",
            "mailto": mailto,
        }
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        raw = http_get(url, mailto)
        if not raw:
            break
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            break
        results = data.get("results", [])
        if not results:
            break
        for w in results:
            best_oa = w.get("best_oa_location") or {}
            ids = w.get("ids") or {}
            out.append({
                "source": "openalex",
                "openalex_id": w.get("id", ""),
                "title": (w.get("title") or "").strip(),
                "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
                "date": w.get("publication_date") or "",
                "year": w.get("publication_year"),
                "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                "arxiv_id": _arxiv_from_ids(ids, w),
                "venue": venue_of(w),
                "type": w.get("type") or "",
                "citations": w.get("cited_by_count") or 0,
                "fwci": w.get("fwci"),
                "percentile": ((w.get("citation_normalized_percentile") or {})
                               .get("value")),
                "cited_by_pct_year": ((w.get("cited_by_percentile_year") or {})
                                      .get("max")),
                "pdf_url": best_oa.get("pdf_url") or "",
                "landing_url": best_oa.get("landing_page_url") or "",
                "is_oa": bool((w.get("open_access") or {}).get("is_oa")),
                "is_retracted": bool(w.get("is_retracted")),
                "is_paratext": bool(w.get("is_paratext")),
                "n_refs": w.get("referenced_works_count") or 0,
                "authors": [((a.get("author") or {}).get("display_name") or "")
                            for a in (w.get("authorships") or [])][:8],
            })
        if len(results) < per_page:
            break
    return out


def _arxiv_from_ids(ids: dict, work: dict) -> str:
    for loc in work.get("locations") or []:
        url = (loc.get("landing_page_url") or "") + " " + (loc.get("pdf_url") or "")
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", url)
        if m:
            return m.group(1)
    for v in ids.values():
        if isinstance(v, str) and "arxiv" in v.lower():
            m = re.search(r"([0-9]{4}\.[0-9]{4,5})", v)
            if m:
                return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Source 2: arXiv
# ---------------------------------------------------------------------------

ATOM = {"a": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str, since: date, until: date,
                 max_results: int = 100) -> list[dict]:
    window = (f"submittedDate:[{since.strftime('%Y%m%d')}0000 "
              f"TO {until.strftime('%Y%m%d')}2359]")
    # arXiv's API does NOT do implicit AND, and a quoted multi-word phrase is an
    # exact-phrase match that almost always returns zero. AND the words instead.
    words = [w for w in re.split(r"\s+", query.strip()) if w and w.upper() != "AND"]
    terms = " AND ".join(f"all:{w}" for w in words) or f"all:{query}"
    params = {
        "search_query": f"({terms}) AND {window}",
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    raw = http_get(url)
    if not raw:
        return []
    try:
        root = ET.fromstring(raw.decode("utf-8", "replace"))
    except ET.ParseError:
        return []
    out: list[dict] = []
    for e in root.findall("a:entry", ATOM):
        def txt(tag: str) -> str:
            node = e.find(f"a:{tag}", ATOM)
            return (node.text or "").strip() if node is not None else ""

        raw_id = txt("id")
        m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})", raw_id)
        aid = m.group(1) if m else ""
        pdf = ""
        for link in e.findall("a:link", ATOM):
            if link.get("title") == "pdf":
                pdf = link.get("href", "")
        published = txt("published")[:10]
        journal_ref = e.find("{http://arxiv.org/schemas/atom}journal_ref")
        out.append({
            "source": "arxiv",
            "openalex_id": "",
            "title": re.sub(r"\s+", " ", txt("title")),
            "abstract": re.sub(r"\s+", " ", txt("summary")),
            "date": published,
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "doi": "",
            "arxiv_id": aid,
            "venue": (journal_ref.text.strip()
                      if journal_ref is not None and journal_ref.text else ""),
            "type": "preprint",
            "citations": 0,
            "fwci": None,
            "percentile": None,
            "cited_by_pct_year": None,
            "pdf_url": pdf or (f"https://arxiv.org/pdf/{aid}" if aid else ""),
            "landing_url": raw_id,
            "is_oa": True,
            "is_retracted": False,
            "is_paratext": False,
            "n_refs": 0,
            "authors": [(a.find("a:name", ATOM).text or "")
                        for a in e.findall("a:author", ATOM)][:8],
        })
    return out


# ---------------------------------------------------------------------------
# Optional enrichment: Semantic Scholar (influential citation count).
# Best-effort — rate-limited without an API key, so failure is expected and fine.
# ---------------------------------------------------------------------------

def enrich_semantic_scholar(papers: list[dict], limit: int = 25) -> int:
    fields = "citationCount,influentialCitationCount,venue,year"
    hits = 0
    for p in papers[:limit]:
        key = f"DOI:{p['doi']}" if p.get("doi") else (
            f"arXiv:{p['arxiv_id']}" if p.get("arxiv_id") else None)
        if not key:
            continue
        url = (f"https://api.semanticscholar.org/graph/v1/paper/"
               f"{urllib.parse.quote(key)}?fields={fields}")
        raw = http_get(url, retries=1, timeout=20)
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if d.get("influentialCitationCount") is not None:
            p["influential_citations"] = d["influentialCitationCount"]
            hits += 1
        if d.get("citationCount") and d["citationCount"] > p.get("citations", 0):
            p["citations"] = d["citationCount"]
        time.sleep(0.4)  # be a good citizen
    return hits


# ---------------------------------------------------------------------------
# Merge / de-duplicate
# ---------------------------------------------------------------------------

def title_key(title: str) -> str:
    return STOP_TITLE_WORDS.sub("", (title or "").lower())[:90]


def dedupe(papers: list[dict]) -> list[dict]:
    """Merge records for the same work. OpenAlex wins on metadata; arXiv
    contributes a guaranteed-downloadable PDF link."""
    by_key: dict[str, dict] = {}
    for p in papers:
        keys = [k for k in (
            f"doi:{p['doi'].lower()}" if p.get("doi") else None,
            f"arx:{p['arxiv_id']}" if p.get("arxiv_id") else None,
            f"ttl:{title_key(p.get('title', ''))}",
        ) if k]
        existing = next((by_key[k] for k in keys if k in by_key), None)
        if existing is None:
            for k in keys:
                by_key[k] = p
            continue
        # merge: prefer richer values
        if p.get("citations", 0) > existing.get("citations", 0):
            existing["citations"] = p["citations"]
        if not existing.get("pdf_url") and p.get("pdf_url"):
            existing["pdf_url"] = p["pdf_url"]
        if not existing.get("arxiv_id") and p.get("arxiv_id"):
            existing["arxiv_id"] = p["arxiv_id"]
        if not existing.get("doi") and p.get("doi"):
            existing["doi"] = p["doi"]
        if not existing.get("abstract") and p.get("abstract"):
            existing["abstract"] = p["abstract"]
        if not existing.get("venue") and p.get("venue"):
            existing["venue"] = p["venue"]
        for k in keys:
            by_key.setdefault(k, existing)
    seen: list[dict] = []
    for v in by_key.values():
        if not any(v is s for s in seen):
            seen.append(v)
    return seen


# ---------------------------------------------------------------------------
# Gates and scoring
# ---------------------------------------------------------------------------

def blob(p: dict) -> str:
    return f"{p.get('title', '')} {p.get('abstract', '')}".lower()


def in_family(p: dict) -> bool:
    """Gate 2 — must actually be about gradient-boosted trees."""
    return any(t in blob(p) for t in GBDT_TERMS)


def family_in_title(p: dict) -> bool:
    return any(t in (p.get("title") or "").lower() for t in GBDT_TERMS)


def months_old(pub: str, today: date) -> float:
    try:
        d = datetime.strptime(pub[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 99.0
    return max((today - d).days / 30.44, 0.1)


def venue_tier(p: dict) -> float:
    v = (p.get("venue") or "").lower()
    if not v:
        return 0.10 if p.get("type") == "preprint" else 0.25
    for score, needles in VENUE_TIERS:
        if any(n in v for n in needles):
            return score
    return 0.30


def course_compliance(p: dict) -> tuple[str, list[str]]:
    """Flag against the 50.007 rule: no deep learning / LLMs as the model.

    Returns (flag, matched_terms) where flag is one of:
      clear    — no neural machinery mentioned; usable as-is
      partial  — neural terms appear, but boosting is present too; usually the
                 boosting half is transferable. Read before trusting.
      blocked  — the paper's headline contribution looks neural
    """
    t = (p.get("title") or "").lower()
    a = (p.get("abstract") or "").lower()
    in_title = [n for n in NEURAL_TERMS if n in t]
    in_abs = [n for n in NEURAL_TERMS if n in a]
    if in_title and not family_in_title(p):
        return "blocked", sorted(set(in_title))
    if in_title or in_abs:
        return "partial", sorted(set(in_title + in_abs))[:6]
    return "clear", []


def topic_relevance(p: dict, topics: dict) -> tuple[float, list[str], list[str]]:
    """Two-tier relevance with an application-domain penalty.

    Returns (points, hits, domains). `points` is 0.0 when no CORE term matched —
    bonus vocabulary alone can never carry a paper through the gate.
    """
    text = blob(p)
    title = (p.get("title") or "").lower()

    core_hits, core_pts = [], 0.0
    for term, weight in topics["core"]:
        if term in text:
            core_pts += weight
            core_hits.append(term)

    if not core_hits:
        return 0.0, [], []

    bonus_pts, bonus_hits = 0.0, []
    for term, weight in topics["bonus"]:
        if term in text:
            bonus_pts += weight
            bonus_hits.append(term)
    # Bonus vocabulary is capped: it should refine a ranking, never drive it.
    bonus_pts = min(bonus_pts, 3.0)

    domains = [d for d in APPLICATION_DOMAINS if d in title]
    total = core_pts + bonus_pts
    if domains:
        # A domain study can still be worth reading if it is *also* squarely
        # about text or about method — halve rather than zero it.
        total *= 0.35

    return total, core_hits + bonus_hits, domains


def impact_score(p: dict, today: date, topics: list[tuple[str, float]]) -> dict:
    """Transparent composite. Every component is reported so the ranking can be
    argued with, not just accepted. See references/impact-scoring.md.

    Two numbers are kept deliberately separate:

      impact_score  — how much the *world* rates this paper (citations, venue).
      relevance     — how much it can plausibly help *your* task.

    They are combined multiplicatively into `final_score`. A 300-citation paper
    on wireless-sensor intrusion detection is genuinely high-impact and
    genuinely useless here; an additive score cannot express that, and ranks it
    third. A multiplicative one drives it toward zero, which is correct.
    """
    age = months_old(p.get("date", ""), today)
    cites = p.get("citations", 0) or 0

    # 1. Citation velocity — cites/month, log-damped. The single most useful
    #    raw signal, but it structurally penalises very recent work.
    velocity = cites / age
    s_velocity = min(math.log1p(velocity * 4) / math.log(21), 1.0)

    # 2. Absolute citations — log-damped, rewards a genuine early hit.
    s_cites = min(math.log1p(cites) / math.log(201), 1.0)

    # 3. Venue quality — the key non-citation signal for young papers.
    s_venue = venue_tier(p)

    # 4. Influential citations (Semantic Scholar), when enrichment succeeded.
    infl = p.get("influential_citations")
    s_infl = min(math.log1p(infl) / math.log(21), 1.0) if infl is not None else None

    # 5. OpenAlex percentile vs same-year works, when computed. The field is a
    #    {min,max} band; taking .max is systematically optimistic (it pins most
    #    cited papers at 1.0), so use the midpoint.
    band = p.get("cited_by_pct_year") or {}
    if isinstance(band, dict) and band.get("min") is not None:
        s_pct = ((band.get("min", 0) + band.get("max", band.get("min", 0))) / 2) / 100.0
    else:
        s_pct = None

    # 6. Accessibility — can we actually read and reproduce it?
    s_access = (0.6 if p.get("pdf_url") else 0.0) + (0.4 if p.get("is_oa") else 0.0)

    # --- impact half: what the world thinks -------------------------------
    parts = {
        "velocity": (s_velocity, 0.30),
        "citations": (s_cites, 0.24),
        "venue": (s_venue, 0.30),
        "access": (s_access, 0.10),
    }
    if s_infl is not None:
        parts["influential"] = (s_infl, 0.14)
    if s_pct is not None:
        parts["percentile"] = (s_pct, 0.14)

    wsum = sum(w for _, w in parts.values())
    impact = sum(v * w for v, w in parts.values()) / wsum

    # --- relevance half: what it is worth to you --------------------------
    topic_raw, topic_hits, domains = topic_relevance(p, topics)
    relevance = min(topic_raw / 10.0, 1.0)
    # Boosting named in the title means it is the subject, not a baseline in a
    # comparison table — the difference between a paper you learn a method from
    # and a paper that merely mentions one.
    if family_in_title(p):
        relevance = min(relevance + 0.15, 1.0)

    # Multiplicative combination, floored so impact is never wholly ignored.
    final = impact * (0.25 + 0.75 * relevance)

    flag, neural_terms = course_compliance(p)
    return {
        "final_score": round(final, 4),
        "impact_score": round(impact, 4),
        "relevance": round(relevance, 4),
        "topic_points": round(topic_raw, 2),
        "age_months": round(age, 1),
        "citations_per_month": round(velocity, 2),
        "components": {k: round(v, 3) for k, (v, _) in parts.items()},
        "topic_hits": topic_hits,
        "application_domains": domains,
        "course_flag": flag,
        "neural_terms": neural_terms,
        "family_in_title": family_in_title(p),
    }


def cohort(age_months: float) -> str:
    if age_months <= 6:
        return "0-6mo (too new to cite — judge on venue/novelty)"
    if age_months <= 12:
        return "6-12mo (citations emerging)"
    return "12-18mo (citations meaningful)"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

FLAG_ICON = {"clear": "OK", "partial": "PARTIAL", "blocked": "BLOCKED"}


def write_shortlist(papers: list[dict], path: Path, since: date, until: date,
                    queries: list[str], stats: dict) -> None:
    s2_note = ("ok" if stats["s2_hits"] else
               "rate-limited or unavailable — scores fall back to OpenAlex signals")
    query_list = "; ".join(queries)
    lines = [
        "# XGBoost / gradient-boosting paper shortlist",
        "",
        f"- **Window:** {since} to {until} "
        f"({stats['months']} months, computed at run time)",
        f"- **Queries:** {len(queries)} — {query_list}",
        f"- **Raw hits:** {stats['raw']} -> **after de-dup:** {stats['deduped']} "
        f"-> **after family gate:** {stats['in_family']} "
        f"-> **shortlisted:** {len(papers)}",
        f"- **Semantic Scholar enrichment:** {stats['s2_hits']} papers ({s2_note})",
        "",
        "Course-compliance flags refer to the 50.007 rule *\"DO NOT use any deep "
        "learning approach (including LLMs)\"*:",
        "`OK` = no neural machinery; `PARTIAL` = neural terms present, boosting "
        "half is usually still transferable; `BLOCKED` = headline contribution "
        "looks neural. **These are keyword heuristics — confirm by reading.**",
        "",
        "> Ranking is a starting point, not a verdict. Read "
        "`references/impact-scoring.md` for what each number can and cannot support.",
        "",
    ]
    by_cohort: dict[str, list[dict]] = {}
    for p in papers:
        by_cohort.setdefault(cohort(p["scoring"]["age_months"]), []).append(p)

    for name in ["12-18mo (citations meaningful)", "6-12mo (citations emerging)",
                 "0-6mo (too new to cite — judge on venue/novelty)"]:
        group = by_cohort.get(name)
        if not group:
            continue
        lines += [f"## Cohort: {name}", "",
                  "| # | Final | Impact | Relev | Flag | Date | Cites | /mo | Venue | Title |",
                  "|---|-------|--------|-------|------|------|-------|-----|-------|-------|"]
        for i, p in enumerate(group, 1):
            s = p["scoring"]
            title = p["title"].replace("|", "\\|")
            link = p.get("landing_url") or p.get("pdf_url") or ""
            title_cell = f"[{title[:70]}]({link})" if link else title[:70]
            venue = (p.get("venue") or "preprint")[:26].replace("|", "\\|")
            lines.append(
                f"| {i} | **{s['final_score']:.3f}** | {s['impact_score']:.3f} "
                f"| {s['relevance']:.2f} | {FLAG_ICON[s['course_flag']]} "
                f"| {p['date']} | {p['citations']} | {s['citations_per_month']:.1f} "
                f"| {venue} | {title_cell} |")
        lines.append("")

    lines += [
        "## Score breakdown (top 10)",
        "",
        "`final = impact x (0.25 + 0.75 x relevance)` — see "
        "`references/impact-scoring.md`.",
        "",
    ]
    for i, p in enumerate(papers[:10], 1):
        s = p["scoring"]
        comps = ", ".join(f"{k}={v}" for k, v in s["components"].items())
        lines += [
            f"**{i}. {p['title']}**  ",
            f"`{p['date']}` · {p.get('venue') or 'preprint'} · "
            f"{p['citations']} citations · final **{s['final_score']:.3f}** "
            f"(impact {s['impact_score']:.3f} x relevance {s['relevance']:.2f})  ",
            f"- impact components: {comps}",
            f"- topic hits ({s['topic_points']} pts): "
            f"{', '.join(s['topic_hits']) or '(none)'}",
            (f"- application-domain penalty applied: "
             f"{', '.join(s['application_domains'])}"
             if s["application_domains"] else
             "- application-domain penalty: none"),
            f"- course flag: **{FLAG_ICON[s['course_flag']]}**"
            + (f" — matched: {', '.join(s['neural_terms'])}" if s["neural_terms"] else ""),
            f"- PDF: {p.get('pdf_url') or '(no open-access PDF found)'}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=18,
                    help="rolling recency window in months (default 18)")
    ap.add_argument("--preset", choices=sorted(PRESET_TOPICS), default="project",
                    help="topic weighting: 'project' biases to AI-text detection, "
                         "'generic' is method-only (default: project)")
    ap.add_argument("--query", action="append", default=[],
                    help="extra search string; repeatable. Adds to the preset.")
    ap.add_argument("--only-query", action="store_true",
                    help="use only --query strings, ignore the preset queries")
    ap.add_argument("--top", type=int, default=20, help="shortlist size (default 20)")
    ap.add_argument("--min-topic", type=float, default=4.0,
                    help="minimum topic-relevance points to survive (default 4.0). "
                         "Raise to tighten, lower to widen. This is the gate that "
                         "keeps out high-citation papers from unrelated domains.")
    ap.add_argument("--out-dir", default="papers", help="output dir (default papers/)")
    ap.add_argument("--mailto", default="anonymous@example.com",
                    help="contact email for the OpenAlex polite pool (faster, kinder)")
    ap.add_argument("--no-arxiv", action="store_true", help="skip arXiv preprints")
    ap.add_argument("--no-s2", action="store_true",
                    help="skip Semantic Scholar enrichment (it is rate-limited)")
    ap.add_argument("--include-blocked", action="store_true",
                    help="keep papers flagged as deep-learning-blocked")
    args = ap.parse_args()

    today = date.today()
    since = today - timedelta(days=int(args.months * 30.44))
    topics = PRESET_TOPICS[args.preset]
    queries = list(args.query) if args.only_query else \
        PRESET_QUERIES[args.preset] + list(args.query)
    if not queries:
        print("no queries — pass --query or drop --only-query", file=sys.stderr)
        return 2

    print(f"[window] {since} .. {today}  ({args.months} months)")
    print(f"[preset] {args.preset} | {len(queries)} queries")

    raw: list[dict] = []
    for q in queries:
        print(f"  openalex: {q!r}")
        got = search_openalex(q, since.isoformat(), today.isoformat(), args.mailto)
        print(f"    -> {len(got)}")
        raw += got
        if not args.no_arxiv:
            print(f"  arxiv   : {q!r}")
            got_a = search_arxiv(q, since, today)
            print(f"    -> {len(got_a)}")
            raw += got_a
            time.sleep(3)  # arXiv asks for ~1 request per 3s

    print(f"[merge] {len(raw)} raw records")
    merged = dedupe(raw)
    print(f"[merge] {len(merged)} unique works")

    kept = [p for p in merged
            if not p.get("is_retracted") and not p.get("is_paratext")
            and in_family(p)]
    print(f"[gate ] {len(kept)} pass the gradient-boosting family gate")
    if not kept:
        print("Nothing survived the family gate. Widen --months or add --query.",
              file=sys.stderr)
        return 1

    s2_hits = 0
    if not args.no_s2:
        print("[s2   ] enriching with influential-citation counts (best effort)")
        prelim = sorted(kept, key=lambda p: p.get("citations", 0), reverse=True)
        s2_hits = enrich_semantic_scholar(prelim, limit=25)
        print(f"[s2   ] enriched {s2_hits}")

    for p in kept:
        p["scoring"] = impact_score(p, today, topics)

    if not args.include_blocked:
        before = len(kept)
        kept = [p for p in kept if p["scoring"]["course_flag"] != "blocked"]
        print(f"[rule ] dropped {before - len(kept)} deep-learning-blocked papers "
              f"(--include-blocked to keep)")

    # Relevance gate. Without it the ranking fills with genuinely high-impact
    # papers from unrelated domains that merely used XGBoost as one baseline.
    before = len(kept)
    kept = [p for p in kept if p["scoring"]["topic_points"] >= args.min_topic]
    print(f"[topic] dropped {before - len(kept)} papers scoring < "
          f"{args.min_topic} topic points (--min-topic to change)")
    if not kept:
        print("Nothing cleared the topic gate. Lower --min-topic or add --query.",
              file=sys.stderr)
        return 1

    kept.sort(key=lambda p: p["scoring"]["final_score"], reverse=True)
    short = kept[:args.top]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(
        json.dumps({"generated": today.isoformat(),
                    "window_start": since.isoformat(),
                    "window_months": args.months,
                    "preset": args.preset,
                    "queries": queries,
                    "papers": kept}, indent=2), encoding="utf-8")
    write_shortlist(short, out / "SHORTLIST.md", since, today, queries,
                    {"raw": len(raw), "deduped": len(merged), "in_family": len(kept),
                     "s2_hits": s2_hits, "months": args.months})

    print(f"\n[done ] {out/'SHORTLIST.md'}  ({len(short)} shortlisted)")
    print(f"[done ] {out/'candidates.json'}  ({len(kept)} scored)")
    print("\nTop 5:")
    for i, p in enumerate(short[:5], 1):
        s = p["scoring"]
        print(f"  {i}. [{s['final_score']:.3f}] imp={s['impact_score']:.2f} "
              f"rel={s['relevance']:.2f} {FLAG_ICON[s['course_flag']]:7s} "
              f"{p['date']} c={p['citations']:<4} {p['title'][:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
