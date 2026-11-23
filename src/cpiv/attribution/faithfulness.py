"""Faithfulness tests: do the cited features actually drive the model's decision?

* Deletion: set the top-k evidence features to their no-change baseline. A faithful
  explanation makes P(cause) drop much more than removing k random features.
* Sufficiency: keep only the top-k evidence features, reset all others. A faithful
  explanation retains much more P(cause) than keeping k random features.
"""

from __future__ import annotations

import numpy as np

from .evidence import occlusion_importance
from .model import CausalAttributor


def _prob(att: CausalAttributor, x: np.ndarray, ci: int) -> float:
    return float(att.predict_proba(x[None, :])[0, ci])


def faithfulness(att: CausalAttributor, X: np.ndarray, Y: np.ndarray | None = None,
                 ks: tuple[int, ...] = (1, 3, 5, 10), n_random: int = 5, max_cases: int = 200,
                 seed: int = 0) -> dict:
    """Evaluate deletion/sufficiency over (sample, cause) cases where the cause is present.

    ``Y`` are ground-truth labels; if omitted, model predictions are used.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    present = np.asarray(Y, int) if Y is not None else att.predict(X)
    cases = [(i, c) for i in range(len(X)) for c in np.flatnonzero(present[i])]
    if not cases:
        return {"n_cases": 0}
    rng.shuffle(cases)
    cases = cases[:max_cases]
    nf = X.shape[1]
    res = {k: {"del_top": [], "del_rand": [], "suf_top": [], "suf_rand": []} for k in ks}
    imp_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, c in cases:
        if i not in imp_cache:
            imp_cache[i] = occlusion_importance(att, X[i])
        p0, imp = imp_cache[i]
        if p0[c] < 0.05:
            continue
        order = np.argsort(-imp[:, c])
        for k in ks:
            top = order[:k]
            x = X[i].copy()
            x[top] = att.baseline_[top]
            res[k]["del_top"].append(p0[c] - _prob(att, x, c))
            xs = att.baseline_.copy()
            xs[top] = X[i][top]
            res[k]["suf_top"].append(_prob(att, xs, c) / p0[c])
            dr, sr = [], []
            for _ in range(n_random):
                r = rng.choice(nf, size=k, replace=False)
                x = X[i].copy()
                x[r] = att.baseline_[r]
                dr.append(p0[c] - _prob(att, x, c))
                xs = att.baseline_.copy()
                xs[r] = X[i][r]
                sr.append(_prob(att, xs, c) / p0[c])
            res[k]["del_rand"].append(np.mean(dr))
            res[k]["suf_rand"].append(np.mean(sr))
    out = {"n_cases": len(cases), "by_k": {}}
    for k in ks:
        r = {m: float(np.mean(v)) if v else float("nan") for m, v in res[k].items()}
        r["deletion_gap"] = r["del_top"] - r["del_rand"]
        r["sufficiency_gap"] = r["suf_top"] - r["suf_rand"]
        out["by_k"][k] = r
    out["mean_deletion_gap"] = float(np.nanmean([out["by_k"][k]["deletion_gap"] for k in ks]))
    out["mean_sufficiency_gap"] = float(np.nanmean([out["by_k"][k]["sufficiency_gap"] for k in ks]))
    return out
