import numpy as np
import pytest

from cpiv.config import CAUSES, UNKNOWN_CAUSES
from cpiv.features.spectral import spectral_features
from cpiv.interventions import REGISTRY, apply_chain, get_intervention


@pytest.mark.parametrize("name", CAUSES + UNKNOWN_CAUSES)
def test_intervention_shape_and_finite(name, voice, rng):
    out, params = get_intervention(name)(voice, 16000, rng, 0.6)
    assert out.shape == voice.shape
    assert np.isfinite(out).all()
    assert params["cause"] == name
    assert not np.allclose(out, voice)


def test_registry_covers_all_causes():
    assert set(CAUSES + UNKNOWN_CAUSES) <= set(REGISTRY)


def test_reproducible_with_same_seed(voice):
    a, _ = get_intervention("additive_noise")(voice, 16000, np.random.default_rng(5), 0.5)
    b, _ = get_intervention("additive_noise")(voice, 16000, np.random.default_rng(5), 0.5)
    assert np.allclose(a, b)


def test_telephone_reduces_bandwidth(voice, rng):
    out, _ = get_intervention("telephone_channel")(voice, 16000, rng, 0.8)
    before, after = spectral_features(voice, 16000), spectral_features(out, 16000)
    # band-pass removes low-frequency energy (the synthetic voice has little energy above 4 kHz)
    assert after["e_0_200"] < before["e_0_200"]
    assert after["e_300_3400"] > before["e_300_3400"]


def test_pitch_shift_moves_f0(voice, rng):
    from cpiv.features.phonetic import phonetic_features

    up = get_intervention("pitch_manipulation")(voice, 16000, rng, params={"semitones": 5.0})[0]
    assert phonetic_features(up, 16000)["f0_median"] > phonetic_features(voice, 16000)["f0_median"]


def test_noise_lowers_snr_proxy(voice, rng):
    out, _ = get_intervention("additive_noise")(voice, 16000, rng, 1.0)
    assert spectral_features(out, 16000)["noise_floor_db"] > spectral_features(voice, 16000)["noise_floor_db"]


def test_chain_logs_each_step(voice, rng):
    _, log = apply_chain(voice, 16000, ["telephone_channel", "additive_noise"], rng, 0.5)
    assert [p["cause"] for p in log] == ["telephone_channel", "additive_noise"]
