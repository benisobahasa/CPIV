"""Score any external system on a built benchmark from a predictions CSV.

CSV format: header ``id,<cause_1>,<cause_2>,...`` with one probability (0-1) per known
cause for every ``id`` in ``manifest.jsonl``. Optional column ``unknown_score`` (higher =
more likely an unknown manipulation) enables open-set scoring.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..config import CAUSES
from .metrics import attribution_metrics, cause_confusion, open_set_metrics


def score_predictions(data_dir: str | Path, predictions_csv: str | Path, threshold: float = 0.5) -> dict:
    meta = [json.loads(line) for line in open(Path(data_dir) / "manifest.jsonl")]
    with open(predictions_csv, newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    missing = [m["id"] for m in meta if m["id"] not in rows]
    if missing:
        raise ValueError(f"{len(missing)} manifest ids missing from predictions, e.g. {missing[:3]}")
    P = np.array([[float(rows[m["id"]][c]) for c in CAUSES] for m in meta])
    Y = np.array([[int(c in m["causes"]) for c in CAUSES] for m in meta])
    splits = np.array([m["split"] for m in meta])
    out: dict = {"protocols": {}}
    for s in sorted(set(splits)):
        if s in ("train", "val", "test_unknown"):
            continue
        idx = splits == s
        out["protocols"][s] = attribution_metrics(P[idx], Y[idx], CAUSES, threshold)
    seen = np.isin(splits, ["test_seen", "test_severity"])
    if seen.any():
        out["confusion"] = cause_confusion(P[seen], Y[seen], CAUSES)
    first = next(iter(rows.values()))
    if "unknown_score" in first:
        u = np.array([float(rows[m["id"]]["unknown_score"]) for m in meta])
        known = np.isin(splits, ["test_seen", "test_severity", "test_combo"])
        unk = splits == "test_unknown"
        out["protocols"]["test_unknown"] = {"novelty": open_set_metrics(u[known], u[unk])}
    return out
