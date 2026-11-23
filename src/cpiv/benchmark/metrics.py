"""Evaluation metrics for multi-label cause attribution and open-set detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def expected_calibration_error(P: np.ndarray, Y: np.ndarray, bins: int = 10) -> float:
    p, y = P.ravel(), Y.ravel()
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def attribution_metrics(P: np.ndarray, Y: np.ndarray, causes: list[str], threshold: float = 0.5) -> dict:
    pred = (P >= threshold).astype(int)
    per = {}
    for i, c in enumerate(causes):
        if Y[:, i].sum() == 0 and pred[:, i].sum() == 0:
            continue
        per[c] = float(f1_score(Y[:, i], pred[:, i], zero_division=0))
    present = Y.sum(0) > 0
    return {
        "n": int(len(Y)),
        "macro_f1": float(np.mean(list(per.values()))) if per else float("nan"),
        "micro_f1": float(f1_score(Y, pred, average="micro", zero_division=0)),
        "subset_accuracy": float(np.mean(np.all(pred == Y, axis=1))),
        "hamming_loss": float(np.mean(pred != Y)),
        "ece": expected_calibration_error(P[:, present], Y[:, present]) if present.any() else float("nan"),
        "per_cause_f1": per,
    }


def open_set_metrics(score_known: np.ndarray, score_unknown: np.ndarray) -> dict:
    """AUROC for detecting unknown manipulations (higher score = more novel)."""
    if len(score_known) == 0 or len(score_unknown) == 0:
        return {"auroc": float("nan")}
    y = np.r_[np.zeros(len(score_known)), np.ones(len(score_unknown))]
    s = np.r_[score_known, score_unknown]
    thr = 1.0
    return {"auroc": float(roc_auc_score(y, s)),
            "fpr_at_novelty_1": float(np.mean(score_known > thr)),
            "tpr_at_novelty_1": float(np.mean(score_unknown > thr))}


def cause_confusion(P: np.ndarray, Y: np.ndarray, causes: list[str]) -> dict[str, dict[str, float]]:
    """For samples with exactly one true cause: mean predicted probability of every cause.

    Off-diagonal mass shows physically overlapping causes (e.g. replay vs. reverberation).
    """
    single = Y.sum(1) == 1
    out: dict[str, dict[str, float]] = {}
    for i, c in enumerate(causes):
        m = single & (Y[:, i] == 1)
        if m.any():
            out[c] = {d: float(P[m, j].mean()) for j, d in enumerate(causes)}
    return out


def top_confusions(conf: dict[str, dict[str, float]], min_prob: float = 0.2, k: int = 8):
    pairs = [(a, b, v) for a, row in conf.items() for b, v in row.items() if a != b and v >= min_prob]
    return sorted(pairs, key=lambda t: -t[2])[:k]
