"""Smoke tests for the pure-math parts (no torch/xgboost needed).

Run from anywhere:  python3 tests/test_smoke.py
"""
import sys
from pathlib import Path

import numpy as np

# The feature extractors live in src/; make them importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features_stylometric import extract_stylometric, extract_stylometric_matrix
from features_surprisal import (diveye_from_surprisal, cpc_from_logprobs,
                                DIVEYE_FEATURE_NAMES)

# ---- stylometric ----
human = ("I honestly wasn't sure this would work?! But we tried it anyway — "
         "my friend and I, at 2am, half-asleep. It failed. Twice. "
         "Then, weirdly, it didn't.")
ai = ("The implementation of the proposed methodology demonstrates "
      "significant improvements across all evaluated metrics. Furthermore, "
      "the results indicate that the approach is robust and generalizable. "
      "In conclusion, the framework provides a comprehensive solution.")

fh = extract_stylometric(human)
fa = extract_stylometric(ai)
assert len(fh) == len(fa) and all(np.isfinite(list(fh.values())))
assert fh["liw_pron_1sg"] > fa["liw_pron_1sg"], "human sample should use more 1sg pronouns"
assert fa["cpx_avg_word_len"] > fh["cpx_avg_word_len"], "AI-ish sample should have longer words"
X, names = extract_stylometric_matrix([human, ai, "", "Hi."])
assert X.shape == (4, len(names)) and np.isfinite(X).all(), "matrix must handle empty/tiny texts"
print(f"stylometric OK: {len(names)} features; e.g. "
      f"1sg pronoun rate human={fh['liw_pron_1sg']:.3f} vs ai={fa['liw_pron_1sg']:.3f}")

# ---- DivEye math ----
rng = np.random.default_rng(0)
# 'machine-like': low-mean, low-variance, smooth surprisal
s_machine = 2.0 + 0.3 * rng.standard_normal(300)
# 'human-like': higher mean/variance, bursty surprisal
s_human = 5.0 + 2.5 * rng.standard_normal(300) + 3.0 * (rng.random(300) < 0.1)

fm = diveye_from_surprisal(s_machine)
fu = diveye_from_surprisal(s_human)
assert fm.shape == (9,) and fu.shape == (9,)
assert fu[0] > fm[0] and fu[1] > fm[1], "human-like should have higher mean/var surprisal"
assert np.isfinite(fm).all() and np.isfinite(fu).all()
assert (diveye_from_surprisal(np.array([1.0, 2.0])) == 0).all(), "short seq -> zeros"
print("DivEye OK:", dict(zip(DIVEYE_FEATURE_NAMES, np.round(fu, 3))))

# ---- CPC math: exact check against a hand-computable 2-token, 3-word vocab case
p1 = np.array([0.7, 0.2, 0.1]); p2 = np.array([0.5, 0.3, 0.2])
lp1, lp2 = np.log(p1), np.log(p2)
mu = np.array([(p1 * lp1).sum(), (p2 * lp2).sum()])
var = np.array([(p1 * lp1**2).sum() - mu[0]**2, (p2 * lp2**2).sum() - mu[1]**2])
tok_lp = np.array([lp1[0], lp2[0]])  # model picked its own argmax both times
cpc, mll = cpc_from_logprobs(tok_lp, mu, var)
expected = (tok_lp.sum() - mu.sum()) / np.sqrt(var.sum())
assert abs(cpc - expected) < 1e-12
assert cpc > 0, "picking the argmax every time must give positive curvature"
# and a 'human-like' choice (low-prob tokens) should score lower
tok_lp_h = np.array([lp1[2], lp2[2]])
cpc_h, _ = cpc_from_logprobs(tok_lp_h, mu, var)
assert cpc_h < cpc
print(f"CPC OK: machine-like={cpc:.3f} > human-like={cpc_h:.3f}")

print("\nALL SMOKE TESTS PASSED")
