"""
Stylometric feature extractor — SKDU / NELA lineage.
=====================================================

Provenance
----------
- SKDU at De-Factify 4.0 (AAAI-25 workshop), arXiv:2503.22338 — showed that
  NELA-toolkit content features fed to XGBoost reach F1 = 0.9979 (dev) /
  0.9945 (test) on binary human-vs-AI detection, significantly outperforming
  RAIDAR rewriting features (F1 = 0.9652).
- NELA toolkit (Horne et al., WWW 2018) — six feature groups (style,
  complexity, bias, affect, moral, event), all normalized by text length.
- LIWC (Pennebaker et al.) is proprietary; this module substitutes open
  LIWC-*like* category counters (pronouns, articles, negations, ...) built
  from public function-word lists. LIWC's own docs warn that scores are
  unreliable below ~25-50 words, so all category features here are
  percentages of total tokens and a `n_tokens` feature is emitted so the
  classifier can learn to discount short texts on its own.

Design constraints honoured
---------------------------
- Zero neural-network dependency: pure Python + numpy. Fully compliant with
  a "no deep learning" course constraint.
- Every feature is normalized by length (NELA's key design decision), which
  makes the vector robust across variable-length documents.
- If the open-source `nela_features` package is installed, you can extract
  the *original* NELA vector as well (see `try_nela_full()`), and
  concatenate both.

Feature groups implemented (names prefixed accordingly):
  sty_*   punctuation / casing / quoting style        (NELA "Style")
  cpx_*   readability & lexical complexity            (NELA "Complexity")
  liw_*   LIWC-like psycholinguistic category rates   (SKDU's LIWC usage)
  frq_*   function-word & frequency profile
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

import numpy as np

# --------------------------------------------------------------------------
# Small open lexicons (LIWC-like categories; public-domain function words).
# --------------------------------------------------------------------------

_PRONOUNS_1SG = {"i", "me", "my", "mine", "myself"}
_PRONOUNS_1PL = {"we", "us", "our", "ours", "ourselves"}
_PRONOUNS_2 = {"you", "your", "yours", "yourself", "yourselves"}
_PRONOUNS_3 = {
    "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs", "himself", "herself",
    "itself", "themselves",
}
_ARTICLES = {"a", "an", "the"}
_NEGATIONS = {
    "no", "not", "never", "none", "nobody", "nothing", "neither",
    "nowhere", "cannot", "cant", "dont", "doesnt", "didnt", "wont",
    "wouldnt", "shouldnt", "couldnt", "isnt", "arent", "wasnt", "werent",
}
_HEDGES = {
    "maybe", "perhaps", "possibly", "probably", "likely", "apparently",
    "seemingly", "somewhat", "arguably", "roughly", "approximately",
    "presumably", "allegedly", "reportedly", "may", "might", "could",
    "suggest", "suggests", "seem", "seems", "appear", "appears",
}
_CERTAINTY = {
    "always", "never", "definitely", "certainly", "undoubtedly",
    "absolutely", "clearly", "obviously", "surely", "must", "every",
    "all", "completely", "totally", "entirely",
}
# A compact stopword list (top English function words) for the frequency
# profile; deliberately small and fixed so the feature space is stable.
_STOPWORDS = {
    "the", "of", "and", "a", "to", "in", "is", "was", "that", "for",
    "it", "with", "as", "his", "on", "be", "at", "by", "i", "this",
    "had", "not", "are", "but", "from", "or", "have", "an", "they",
    "which", "one", "you", "were", "her", "all", "she", "there",
    "would", "their", "we", "him", "been", "has", "when", "who",
    "will", "more", "no", "if", "out", "so", "said", "what", "up",
    "its", "about", "into", "than", "them", "can", "only", "other",
}

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[.!?]+(?:\s|$)")
_VOWELS = "aeiouy"


def _syllables(word: str) -> int:
    """Cheap syllable estimate (standard heuristic used by readability
    formulas when no dictionary is available)."""
    w = word.lower().strip(string.punctuation)
    if not w:
        return 0
    count, prev_vowel = 0, False
    for ch in w:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _rate(count: float, total: float) -> float:
    return count / total if total > 0 else 0.0


def extract_stylometric(text: str) -> dict[str, float]:
    """Return an ordered dict of stylometric features for one document.

    All rate features are per-token or per-character so that the vector is
    length-normalized (NELA's design). `n_tokens` and `n_sents` are included
    raw so the model can condition on document size (LIWC short-text caveat).
    """
    feats: dict[str, float] = {}

    chars = len(text)
    tokens = _WORD_RE.findall(text)
    tokens_lower = [t.lower() for t in tokens]
    n_tok = len(tokens)
    sents = [s for s in _SENT_RE.split(text) if s.strip()]
    n_sent = max(len(sents), 1)

    feats["n_tokens"] = float(n_tok)
    feats["n_sents"] = float(n_sent)

    # ---------------- Style (punctuation / casing / quoting) --------------
    punct_counts = Counter(c for c in text if c in string.punctuation)
    for p, name in [
        (",", "comma"), (".", "period"), ("!", "exclam"), ("?", "question"),
        (";", "semicolon"), (":", "colon"), ("-", "dash"), ("(", "paren"),
        ('"', "dquote"), ("'", "squote"),
    ]:
        feats[f"sty_{name}_rate"] = _rate(punct_counts.get(p, 0), chars)
    feats["sty_punct_total_rate"] = _rate(sum(punct_counts.values()), chars)
    feats["sty_allcaps_rate"] = _rate(
        sum(1 for t in tokens if len(t) > 1 and t.isupper()), n_tok)
    feats["sty_capitalized_rate"] = _rate(
        sum(1 for t in tokens if t[:1].isupper()), n_tok)
    feats["sty_digit_rate"] = _rate(sum(c.isdigit() for c in text), chars)

    # ---------------- Complexity / readability -----------------------------
    types = set(tokens_lower)
    feats["cpx_ttr"] = _rate(len(types), n_tok)  # type-token ratio |V|/N
    # Herdan's C = log|V| / logN  (length-robust lexical diversity)
    feats["cpx_herdan_c"] = (
        math.log(len(types)) / math.log(n_tok) if n_tok > 1 and types else 0.0
    )
    word_lens = [len(t) for t in tokens] or [0]
    feats["cpx_avg_word_len"] = float(np.mean(word_lens))
    feats["cpx_std_word_len"] = float(np.std(word_lens))
    feats["cpx_avg_sent_len"] = _rate(n_tok, n_sent)
    sent_lens = [len(_WORD_RE.findall(s)) for s in sents] or [0]
    feats["cpx_std_sent_len"] = float(np.std(sent_lens))

    syl = [_syllables(t) for t in tokens] or [0]
    total_syl = sum(syl)
    poly = sum(1 for s in syl if s >= 3)
    if n_tok > 0:
        # Flesch Reading Ease: 206.835 - 1.015*(W/S) - 84.6*(Syl/W)
        feats["cpx_flesch"] = (
            206.835 - 1.015 * (n_tok / n_sent) - 84.6 * (total_syl / n_tok)
        )
        # Flesch-Kincaid Grade: 0.39*(W/S) + 11.8*(Syl/W) - 15.59
        feats["cpx_fk_grade"] = (
            0.39 * (n_tok / n_sent) + 11.8 * (total_syl / n_tok) - 15.59
        )
        # Gunning Fog: 0.4*((W/S) + 100*(complex/W))
        feats["cpx_fog"] = 0.4 * ((n_tok / n_sent) + 100.0 * poly / n_tok)
        # SMOG (approx): 1.043*sqrt(poly*30/S) + 3.1291
        feats["cpx_smog"] = 1.043 * math.sqrt(poly * 30.0 / n_sent) + 3.1291
    else:
        feats["cpx_flesch"] = feats["cpx_fk_grade"] = 0.0
        feats["cpx_fog"] = feats["cpx_smog"] = 0.0
    feats["cpx_long_word_rate"] = _rate(
        sum(1 for t in tokens if len(t) >= 7), n_tok)

    # ---------------- LIWC-like psycholinguistic categories ---------------
    tl_counter = Counter(tokens_lower)

    def cat_rate(cat: set[str]) -> float:
        return _rate(sum(tl_counter[w] for w in cat), n_tok)

    feats["liw_pron_1sg"] = cat_rate(_PRONOUNS_1SG)
    feats["liw_pron_1pl"] = cat_rate(_PRONOUNS_1PL)
    feats["liw_pron_2"] = cat_rate(_PRONOUNS_2)
    feats["liw_pron_3"] = cat_rate(_PRONOUNS_3)
    feats["liw_articles"] = cat_rate(_ARTICLES)
    feats["liw_negations"] = cat_rate(_NEGATIONS)
    feats["liw_hedges"] = cat_rate(_HEDGES)
    feats["liw_certainty"] = cat_rate(_CERTAINTY)

    # ---------------- Function-word / frequency profile --------------------
    stop_hits = sum(tl_counter[w] for w in _STOPWORDS)
    feats["frq_stopword_rate"] = _rate(stop_hits, n_tok)
    # hapax legomena rate — words used exactly once (humans re-use less)
    feats["frq_hapax_rate"] = _rate(
        sum(1 for _, c in tl_counter.items() if c == 1), len(types) or 1)
    # top-10 token mass — how concentrated the word distribution is
    top10 = sum(c for _, c in tl_counter.most_common(10))
    feats["frq_top10_mass"] = _rate(top10, n_tok)
    # Shannon entropy of the unigram distribution (bits/token)
    if n_tok > 0:
        probs = np.array(list(tl_counter.values()), dtype=float) / n_tok
        feats["frq_unigram_entropy"] = float(-(probs * np.log2(probs)).sum())
    else:
        feats["frq_unigram_entropy"] = 0.0

    return feats


FEATURE_NAMES: list[str] = list(extract_stylometric("Sample text. Yes!").keys())


def extract_stylometric_matrix(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Vectorize a corpus. Returns (X, feature_names)."""
    rows = [extract_stylometric(t) for t in texts]
    X = np.array([[r[k] for k in FEATURE_NAMES] for r in rows], dtype=np.float64)
    return X, list(FEATURE_NAMES)


def try_nela_full(texts: list[str]):
    """Optionally extract the *original* NELA feature vector if the
    open-source `nela_features` package is installed (pip install
    nela_features). Returns (X, names) or (None, None) if unavailable.

    SKDU used the full NELA toolkit; concatenating this with the custom
    features above is the closest reproduction of their winning setup.
    """
    try:
        from nela_features.nela_features import NELAFeatureExtractor
    except ImportError:
        return None, None
    nela = NELAFeatureExtractor()
    vecs, names = [], None
    for t in texts:
        vec, names = nela.extract_all(t)
        vecs.append(vec)
    return np.array(vecs, dtype=np.float64), list(names)
