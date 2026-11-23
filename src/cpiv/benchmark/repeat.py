"""Repeated runs over seeds, modes and backends with mean, std and 95% confidence intervals."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .generate import build_dataset
from .run import train_and_evaluate

METRICS = {
    "seen micro-F1": ("test_seen", "micro_f1"),
    "seen macro-F1": ("test_seen", "macro_f1"),
    "severity micro-F1": ("test_severity", "micro_f1"),
    "combo micro-F1": ("test_combo", "micro_f1"),
    "combo subset acc": ("test_combo", "subset_accuracy"),
}


def _extract(res: dict) -> dict[str, float]:
    out = {}
    for label, (proto, key) in METRICS.items():
        v = res["protocols"].get(proto, {}).get(key)
        if v is not None:
            out[label] = float(v)
    u = res["protocols"].get("test_unknown", {}).get("novelty", {}).get("auroc")
    if u is not None:
        out["unknown novelty AUROC"] = float(u)
    f = res.get("faithfulness", {})
    if f.get("n_cases"):
        out["deletion gap"] = f["mean_deletion_gap"]
        out["sufficiency gap"] = f["mean_sufficiency_gap"]
    return out


def _ci(values: list[float]) -> tuple[float, float, float]:
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(a) == 1:
        return float(a[0]), 0.0, 0.0
    sd = float(a.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(a) - 1) * sd / np.sqrt(len(a)))
    return float(a.mean()), sd, half


def repeat_benchmark(items: list[dict], seeds: list[int], modes=("paired",), backends=("rf",),
                     n_estimators: int = 200, n_jobs: int = -1, **build_kwargs) -> dict:
    """Rebuild the benchmark for every seed (new speaker split and new interventions) and
    retrain every (mode, backend) combination. Returns aggregated statistics."""
    runs: dict[str, list[dict[str, float]]] = {}
    for seed in seeds:
        paired = "paired" in modes
        data = build_dataset(items, paired=paired, seed=seed, n_jobs=n_jobs, verbose=False, **build_kwargs)
        for mode in modes:
            for backend in backends:
                res = train_and_evaluate(data, mode=mode, backend=backend, seed=seed,
                                         n_estimators=n_estimators)
                runs.setdefault(f"{mode}/{backend}", []).append(_extract(res))
    summary = {}
    for key, lst in runs.items():
        summary[key] = {}
        for m in sorted({k for r in lst for k in r}):
            mean, sd, half = _ci([r[m] for r in lst if m in r])
            summary[key][m] = {"mean": mean, "std": sd, "ci95": half, "n": len(lst)}
    return {"seeds": list(seeds), "summary": summary, "runs": runs}


def repeat_markdown(rep: dict) -> str:
    L = [f"# CPIV repeated benchmark (seeds: {rep['seeds']})", ""]
    for key, metrics in rep["summary"].items():
        L += [f"## {key}", "", "| Metric | mean | std | 95% CI (+/-) | runs |", "|---|---|---|---|---|"]
        for m, v in metrics.items():
            L.append(f"| {m} | {v['mean']:.3f} | {v['std']:.3f} | {v['ci95']:.3f} | {v['n']} |")
        L.append("")
    return "\n".join(L)
