import csv

import numpy as np
import pytest

from cpiv.attribution import CausalAttributor, explain, faithfulness
from cpiv.benchmark import (
    build_dataset,
    evaluate_model,
    repeat_benchmark,
    repeat_markdown,
    score_predictions,
    split_indices,
    synthetic_sources,
    train_and_evaluate,
)
from cpiv.benchmark.metrics import cause_confusion, top_confusions
from cpiv.config import CAUSES


def test_cause_confusion_shapes():
    Y = np.eye(len(CAUSES), dtype=int)
    P = np.clip(Y + 0.1, 0, 1)
    conf = cause_confusion(P, Y, CAUSES)
    assert set(conf) == set(CAUSES)
    assert all(conf[c][c] > 0.9 for c in CAUSES)
    assert top_confusions(conf, min_prob=0.5) == []


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("extras")
    data = build_dataset(synthetic_sources(6, 1, seed=4, duration=2.0), out_dir=out, per_cause=1,
                         n_pairs=2, seed=4, n_jobs=2, verbose=False)
    return data, out


def test_score_external_predictions(tiny, tmp_path):
    data, out = tiny
    sp = split_indices(data)
    assert len(sp["test_seen"]) > 0
    path = tmp_path / "pred.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", *CAUSES, "unknown_score"])
        for m, y in zip(data["meta"], data["Y"], strict=True):
            w.writerow([m["id"], *[float(v) for v in y], 2.0 if m["unknown"] else 0.1])
    res = score_predictions(out, path)
    assert res["protocols"]["test_seen"]["micro_f1"] == pytest.approx(1.0)  # oracle predictions
    assert res["protocols"]["test_unknown"]["novelty"]["auroc"] == pytest.approx(1.0)


def test_score_detects_missing_ids(tiny, tmp_path):
    _, out = tiny
    path = tmp_path / "bad.csv"
    path.write_text("id," + ",".join(CAUSES) + "\n")
    with pytest.raises(ValueError):
        score_predictions(out, path)


def test_evaluate_model_and_roundtrip(tiny, tmp_path):
    data, _ = tiny
    res = train_and_evaluate(data, n_estimators=30, run_faithfulness=False,
                             model_path=tmp_path / "m.joblib")
    att = CausalAttributor.load(tmp_path / "m.joblib")
    ev = evaluate_model(att, data)
    assert "test_seen" in ev["splits"]
    assert "confusion" in res
    assert faithfulness(att, data["X"][:10], data["Y"][:10], max_cases=5)["n_cases"] >= 0
    assert explain(att, data["X"][0]).to_dict()["causes"]


def test_repeat_benchmark_statistics():
    rep = repeat_benchmark(synthetic_sources(5, 1, seed=1, duration=2.0), seeds=[0, 1], modes=("paired",),
                           backends=("logreg",), n_estimators=20, n_jobs=2, per_cause=1, n_pairs=2)
    s = rep["summary"]["paired/logreg"]["seen micro-F1"]
    assert s["n"] == 2 and 0 <= s["mean"] <= 1
    assert "95% CI" in repeat_markdown(rep)


def test_collect_sources_group_depth_and_pattern(tmp_path):
    from cpiv.audio import save_audio, synth_voice
    from cpiv.benchmark import collect_sources

    y = synth_voice(1, duration=0.5)
    for spk in ("100", "200"):
        for chap in ("a", "b"):
            save_audio(tmp_path / spk / chap / f"{spk}_{chap}_mic1.wav", y)
            save_audio(tmp_path / spk / chap / f"{spk}_{chap}_mic2.wav", y)
    by_parent = collect_sources(tmp_path)
    assert {i["group"] for i in by_parent} == {"a", "b"}          # chapter folders (wrong for LibriSpeech)
    by_speaker = collect_sources(tmp_path, group_depth=1)
    assert {i["group"] for i in by_speaker} == {"100", "200"}
    only_mic1 = collect_sources(tmp_path, pattern="*_mic1.wav", group_depth=1)
    assert len(only_mic1) == 4
