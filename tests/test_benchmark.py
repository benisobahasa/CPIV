import numpy as np
import pytest

from cpiv.benchmark import build_dataset, split_indices, synthetic_sources, train_and_evaluate
from cpiv.benchmark.generate import assign_source_splits
from cpiv.benchmark.spec import conditions_for_source, heldout_pairs
from cpiv.config import CAUSES, TRAIN_PAIRS


def test_source_splits_are_group_disjoint():
    items = synthetic_sources(10, 3, seed=0)
    m = assign_source_splits(items)
    assert set(m.values()) == {"train", "val", "test"}
    assert len(m) == 10


def test_heldout_pairs_exclude_training_pairs():
    hp = {tuple(sorted(p)) for p in heldout_pairs()}
    assert not hp & {tuple(sorted(p)) for p in TRAIN_PAIRS}


def test_test_conditions_contain_all_protocols():
    conds = conditions_for_source("test", np.random.default_rng(0))
    assert {c.split for c in conds} == {"test_seen", "test_severity", "test_combo", "test_unknown"}
    assert min(c.severity for c in conds if c.split == "test_severity") >= 0.8


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    items = synthetic_sources(6, 1, seed=2, duration=2.0)
    return build_dataset(items, out_dir=out, per_cause=1, n_pairs=2, n_unknown=1, seed=2,
                         n_jobs=2, verbose=False), out


def test_build_dataset_outputs(tiny):
    data, out = tiny
    assert (out / "manifest.jsonl").exists() and (out / "features.npz").exists()
    assert data["Y"].shape[1] == len(CAUSES)
    sp = split_indices(data)
    assert len(sp["train"]) > 0 and len(sp["test_seen"]) > 0
    # no source utterance appears in both train and any test split
    train_groups = {data["meta"][i]["group"] for i in sp["train"]}
    test_groups = {data["meta"][i]["group"] for s, ix in sp.items() if s.startswith("test") for i in ix}
    assert not train_groups & test_groups


def test_train_and_evaluate_end_to_end(tiny):
    data, _ = tiny
    res = train_and_evaluate(data, n_estimators=40, run_faithfulness=True)
    assert "test_seen" in res["protocols"]
    assert 0.0 <= res["protocols"]["test_seen"]["micro_f1"] <= 1.0
    att = res["_model"]
    x = data["X"][0]
    from cpiv.attribution import explain

    r = explain(att, x, top_k=3)
    assert len(r.causes) == len(CAUSES)
    assert isinstance(r.text(), str) and r.to_dict()["causes"]
