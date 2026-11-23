import numpy as np
import pytest

from cpiv.audio import synth_voice


@pytest.fixture(scope="session")
def voice():
    return synth_voice(seed=3, duration=2.0)


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
