import numpy as np

from cpiv.features import feature_vector, group_of


def test_paired_vector_layout(voice):
    x, names = feature_vector(voice, voice)
    assert len(x) == len(names) and len(names) % 2 == 0
    half = len(names) // 2
    assert all(n.startswith("delta::") for n in names[:half])
    assert np.allclose(x[:half], 0.0)  # identical signals -> zero delta
    assert np.isfinite(x).all()


def test_blind_vector_layout(voice):
    _, names = feature_vector(voice)
    assert all(n.startswith("test::") for n in names)


def test_groups_cover_everything(voice):
    _, names = feature_vector(voice)
    assert all(group_of(n) != "other" for n in names)
