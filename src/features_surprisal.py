"""
Surprisal-diversity + curvature features — DivEye / Fast-DetectGPT lineage.
===========================================================================

Provenance (verified against primary sources)
---------------------------------------------
- DivEye ("Diversity Boosts AI-Generated Text Detection", Basani & Chen,
  accepted to TMLR 2026, arXiv:2509.18880). Defines a 9-dimensional
  feature vector over the token surprisal sequence and feeds it to a
  lightweight XGBoost meta-classifier. The **corrected** composition
  (per the paper, not the earlier simplified summary) is:

      Distribution (4): mean, variance, skewness, kurtosis of surprisal
      1st-order   (2): mean and variance of first-order differences
      2nd-order   (3): variance, entropy, autocorrelation of
                        second-order differences

- Fast-DetectGPT (Bao et al., ICLR 2024, arXiv:2310.05130). Introduces
  *conditional probability curvature*, computed analytically from the
  model's per-position token distributions (no perturbation sampling),
  which is what makes it ~340x faster than DetectGPT. NotAI.AI
  (arXiv:2603.05617, unreviewed preprint) uses exactly this quantity as
  one interpretable feature in an XGBoost meta-classifier — the design
  this module reproduces.

Mathematics
-----------
Given a document x = (x_1..x_T) and a frozen autoregressive LM p:

  Surprisal:      s_t = -log2 p(x_t | x_{<t})
  1st-order diff: d1_t = s_{t+1} - s_t
  2nd-order diff: d2_t = d1_{t+1} - d1_t

DivEye vector [9]:
  f1 = mean(s)          f2 = var(s)
  f3 = skew(s)          f4 = kurtosis(s)         (Fisher, excess)
  f5 = mean(d1)         f6 = var(d1)
  f7 = var(d2)
  f8 = H(d2)            Shannon entropy of the histogram of d2 (bits)
  f9 = rho_1(d2)        lag-1 autocorrelation of d2

Fast-DetectGPT conditional probability curvature (analytic, same-model
scoring — the "sampling-free" estimator from the official repo):

           sum_t log p(x_t|x_{<t})  -  sum_t mu_t
  CPC  =  ---------------------------------------- ,
                    sqrt( sum_t sigma^2_t )

  where over the full vocabulary V at position t:
    mu_t      = E_{v~p(.|x_{<t})}[ log p(v|x_{<t}) ]
              = sum_v p(v|x_{<t}) log p(v|x_{<t})
    sigma^2_t = Var[ log p(v|x_{<t}) ]
              = sum_v p(v) log^2 p(v) - mu_t^2

  Machine text tends to have CPC >> 0 (the model finds its own choices
  unusually probable relative to its conditional expectation); human
  text sits near/below 0.

Backbone choice
---------------
GPT-2 small (124M) is the cheapest workable backbone (CPU-friendly,
~0.01 s/sample per DivEye's reported extraction speed). DivEye also
validated Llama-3.1-8B / Mistral-7B backbones with comparable results,
so use `model_name` to swap. NOTE: this entire module requires a frozen
LM for *feature extraction only* — the classifier stays XGBoost. If your
course forbids any neural component, disable this module and use the
stylometric features alone (SKDU showed those alone reach F1≈0.99+).

The pure-math functions (`diveye_from_surprisal`, `cpc_from_logprobs`)
are separated from the LM plumbing so they are unit-testable without
torch/transformers installed.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

DIVEYE_FEATURE_NAMES = [
    "div_surp_mean", "div_surp_var", "div_surp_skew", "div_surp_kurt",
    "div_d1_mean", "div_d1_var",
    "div_d2_var", "div_d2_entropy", "div_d2_autocorr",
]
CPC_FEATURE_NAMES = ["fdg_cpc", "fdg_loglik_mean"]


# --------------------------------------------------------------------------
# Pure math — no torch required.
# --------------------------------------------------------------------------

def _hist_entropy_bits(x: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy (bits) of the empirical histogram of x."""
    if x.size < 2:
        return 0.0
    counts, _ = np.histogram(x, bins=bins)
    p = counts.astype(float)
    p = p[p > 0]
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def _lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation rho_1 = Cov(x_t, x_{t+1}) / Var(x)."""
    if x.size < 3:
        return 0.0
    x0, x1 = x[:-1], x[1:]
    sd0, sd1 = x0.std(), x1.std()
    if sd0 == 0 or sd1 == 0:
        return 0.0
    return float(((x0 - x0.mean()) * (x1 - x1.mean())).mean() / (sd0 * sd1))


def diveye_from_surprisal(s: np.ndarray) -> np.ndarray:
    """Compute the 9-dim DivEye vector from a surprisal sequence s_t.

    Uses Fisher (excess) kurtosis and bias-corrected skew, matching
    scipy defaults; degenerate/short sequences fall back to zeros so the
    pipeline never crashes on 1-2 token inputs.
    """
    s = np.asarray(s, dtype=np.float64)
    if s.size < 4:
        return np.zeros(9)
    d1 = np.diff(s, n=1)
    d2 = np.diff(s, n=2)
    return np.array([
        s.mean(),
        s.var(),
        float(sps.skew(s)) if s.std() > 0 else 0.0,
        float(sps.kurtosis(s)) if s.std() > 0 else 0.0,   # excess kurtosis
        d1.mean(),
        d1.var(),
        d2.var(),
        _hist_entropy_bits(d2),
        _lag1_autocorr(d2),
    ])


def cpc_from_logprobs(token_logprobs: np.ndarray,
                      mu_t: np.ndarray,
                      var_t: np.ndarray) -> tuple[float, float]:
    """Analytic Fast-DetectGPT conditional probability curvature.

    Parameters
    ----------
    token_logprobs : log p(x_t | x_{<t}) for the observed tokens, shape (T,)
    mu_t           : per-position expected log-prob over the vocab, shape (T,)
    var_t          : per-position variance of log-prob over the vocab, (T,)

    Returns (cpc, mean_loglik).
    """
    token_logprobs = np.asarray(token_logprobs, dtype=np.float64)
    mu_t = np.asarray(mu_t, dtype=np.float64)
    var_t = np.asarray(var_t, dtype=np.float64)
    denom = np.sqrt(max(var_t.sum(), 1e-12))
    cpc = float((token_logprobs.sum() - mu_t.sum()) / denom)
    return cpc, float(token_logprobs.mean())


# --------------------------------------------------------------------------
# LM plumbing — requires torch + transformers. Lazily imported.
# --------------------------------------------------------------------------

class SurprisalExtractor:
    """Frozen-LM feature extractor producing DivEye [9] + CPC [2] features.

    Usage:
        ext = SurprisalExtractor("gpt2")           # or "gpt2-medium", ...
        X, names = ext.extract_matrix(list_of_texts)
    """

    def __init__(self, model_name: str = "gpt2", device: str | None = None,
                 max_length: int = 1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.max_length = max_length

    @property
    def feature_names(self) -> list[str]:
        return DIVEYE_FEATURE_NAMES + CPC_FEATURE_NAMES

    def _forward(self, text: str):
        """One forward pass; returns (token_logprobs, mu_t, var_t) as numpy.

        log-softmax is computed in float64 on CPU to keep the vocabulary
        sums numerically stable.
        """
        torch = self.torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.max_length)
        ids = enc.input_ids.to(self.device)
        if ids.shape[1] < 2:
            return None
        with torch.no_grad():
            logits = self.model(ids).logits[0]          # (T, V)
        # Positions 0..T-2 predict tokens 1..T-1
        logp = torch.log_softmax(logits[:-1].double(), dim=-1)   # (T-1, V)
        targets = ids[0, 1:]                                     # (T-1,)
        tok_lp = logp.gather(1, targets.unsqueeze(1)).squeeze(1)  # (T-1,)
        p = logp.exp()
        mu = (p * logp).sum(dim=-1)                               # (T-1,)
        var = (p * logp.pow(2)).sum(dim=-1) - mu.pow(2)           # (T-1,)
        return (tok_lp.cpu().numpy(), mu.cpu().numpy(), var.cpu().numpy())

    def extract_one(self, text: str) -> np.ndarray:
        out = self._forward(text)
        if out is None:
            return np.zeros(len(self.feature_names))
        tok_lp, mu, var = out
        # DivEye surprisal is in bits: s_t = -log2 p = -logp / ln 2
        surprisal_bits = -tok_lp / np.log(2.0)
        div = diveye_from_surprisal(surprisal_bits)
        cpc, mean_ll = cpc_from_logprobs(tok_lp, mu, var)
        return np.concatenate([div, [cpc, mean_ll]])

    def extract_matrix(self, texts: list[str],
                       progress: bool = True) -> tuple[np.ndarray, list[str]]:
        rows = []
        for i, t in enumerate(texts):
            rows.append(self.extract_one(t))
            if progress and (i + 1) % 200 == 0:
                print(f"  surprisal features: {i + 1}/{len(texts)}")
        return np.vstack(rows), self.feature_names
