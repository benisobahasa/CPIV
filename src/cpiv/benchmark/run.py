"""Train on the train split, evaluate on every benchmark protocol, write a report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..attribution import CausalAttributor, faithfulness
from ..config import CAUSES
from .generate import select_mode, split_indices
from .metrics import attribution_metrics, cause_confusion, open_set_metrics, top_confusions

PROTOCOLS = {
    "test_seen": "P1 seen causes, seen severity, unseen speakers/utterances",
    "test_severity": "P2 seen causes at stronger severity than training",
    "test_combo": "P3 held-out cause combinations never seen in training",
    "test_unknown": "P4 manipulations outside the known cause list (open-set)",
}


def train_and_evaluate(data: dict, mode: str = "paired", backend: str = "rf", seed: int = 0,
                       n_estimators: int = 300, run_faithfulness: bool = True,
                       model_path: str | Path | None = None) -> dict:
    X, names = select_mode(data, mode)
    Y = data["Y"]
    sp = split_indices(data)
    tr = sp["train"]
    att = CausalAttributor(backend=backend, n_estimators=n_estimators, random_state=seed)
    att.fit(X[tr], Y[tr], names, X_val=X[sp["val"]] if len(sp.get("val", [])) else None)
    if model_path:
        att.save(model_path)

    results: dict = {"mode": mode, "backend": backend, "n_features": len(names),
                     "n_train": int(len(tr)), "protocols": {}}
    for name in ("test_seen", "test_severity", "test_combo"):
        idx = sp.get(name, np.array([], int))
        if len(idx):
            P = att.predict_proba(X[idx])
            r = attribution_metrics(P, Y[idx], CAUSES, att.threshold)
            r["mean_novelty"] = float(att.novelty(X[idx]).mean())
            results["protocols"][name] = r

    known_idx = np.concatenate([sp[s] for s in ("test_seen", "test_severity", "test_combo") if s in sp and len(sp[s])]
                               or [np.array([], int)])
    unk_idx = sp.get("test_unknown", np.array([], int))
    if len(known_idx) and len(unk_idx):
        results["protocols"]["test_unknown"] = {
            "n": int(len(unk_idx)),
            "novelty": open_set_metrics(att.novelty(X[known_idx]), att.novelty(X[unk_idx])),
            "uncertainty": open_set_metrics(att.uncertainty(X[known_idx]), att.uncertainty(X[unk_idx])),
        }

    conf_idx = np.concatenate([sp[k] for k in ("test_seen", "test_severity") if k in sp and len(sp[k])]
                              or [np.array([], int)])
    if len(conf_idx):
        results["confusion"] = cause_confusion(att.predict_proba(X[conf_idx]), Y[conf_idx], CAUSES)

    if run_faithfulness and len(sp.get("test_seen", [])):
        idx = sp["test_seen"]
        results["faithfulness"] = faithfulness(att, X[idx], Y[idx], max_cases=120, seed=seed)
    results["_model"] = att
    return results


def to_markdown(res: dict) -> str:
    L = [f"# CPIV benchmark report ({res['mode']} / {res['backend']})", "",
         f"Features: {res['n_features']}, training samples: {res['n_train']}", "",
         "| Protocol | n | macro-F1 | micro-F1 | subset acc | Hamming | ECE |",
         "|---|---|---|---|---|---|---|"]
    for k in ("test_seen", "test_severity", "test_combo"):
        r = res["protocols"].get(k)
        if r:
            L.append(f"| {k} | {r['n']} | {r['macro_f1']:.3f} | {r['micro_f1']:.3f} | "
                     f"{r['subset_accuracy']:.3f} | {r['hamming_loss']:.3f} | {r['ece']:.3f} |")
    u = res["protocols"].get("test_unknown")
    if u:
        L += ["", f"**Open-set (unknown manipulations, n={u['n']})**", "",
              f"- novelty AUROC: {u['novelty']['auroc']:.3f} "
              f"(TPR {u['novelty']['tpr_at_novelty_1']:.2f} / FPR {u['novelty']['fpr_at_novelty_1']:.2f} at novelty>1)",
              f"- uncertainty AUROC: {u['uncertainty']['auroc']:.3f}"]
    f = res.get("faithfulness")
    if f and f.get("n_cases"):
        L += ["", f"**Faithfulness** ({f['n_cases']} cases): mean deletion gap "
              f"{f['mean_deletion_gap']:+.3f}, mean sufficiency gap {f['mean_sufficiency_gap']:+.3f} "
              "(positive = evidence features matter more than random ones)"]
    conf = top_confusions(res.get("confusion", {}))
    if conf:
        L += ["", "**Most confused causes** (true cause -> mean probability given to another cause):", ""]
        L += [f"- {a} -> {b}: {v:.2f}" for a, b, v in conf]
    L += ["", "Per-cause F1 (test_seen):", ""]
    for c, v in res["protocols"].get("test_seen", {}).get("per_cause_f1", {}).items():
        L.append(f"- {c}: {v:.3f}")
    return "\n".join(L) + "\n"


def save_report(res: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in res.items() if k != "_model"}
    (out / "results.json").write_text(json.dumps(clean, indent=2, default=float))
    (out / "report.md").write_text(to_markdown(res))


def evaluate_model(att: CausalAttributor, data: dict) -> dict:
    """Evaluate a trained model on any built dataset (e.g. a different corpus = cross-corpus test).

    Features are matched to the model by name, so paired/blind layouts must agree.
    """
    names = data["feature_names"]
    missing = [n for n in att.feature_names if n not in names]
    if missing:
        raise ValueError(f"dataset lacks {len(missing)} features the model needs (paired vs blind mismatch?)")
    cols = [names.index(n) for n in att.feature_names]
    X, Y = data["X"][:, cols], data["Y"]
    out: dict = {"splits": {}}
    sp = split_indices(data)
    for name, idx in sp.items():
        if name == "test_unknown" or not len(idx):
            continue
        P = att.predict_proba(X[idx])
        r = attribution_metrics(P, Y[idx], CAUSES, att.threshold)
        r["mean_novelty"] = float(att.novelty(X[idx]).mean())
        out["splits"][name] = r
    return out
