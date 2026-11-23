"""Room and playback/record chain interventions: reverberation and replay."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .base import Intervention, jitter_severity, lerp


def synth_rir(sr: int, rt60: float, rng: np.random.Generator, direct_delay_ms: float = 5.0) -> np.ndarray:
    """Exponentially decaying noise RIR with a direct-path impulse (statistical model)."""
    n = int(sr * max(rt60 * 1.2, 0.05))
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)
    # frequency dependent decay: highs die faster
    b, a = signal.butter(1, 4000 / (sr / 2), "low")
    tail = signal.lfilter(b, a, tail)
    rir = tail * 0.15
    d = int(direct_delay_ms * 1e-3 * sr)
    rir[d] += 1.0
    return rir / (np.max(np.abs(rir)) + 1e-12)


def convolve_rir(y: np.ndarray, rir: np.ndarray) -> np.ndarray:
    out = signal.fftconvolve(y, rir)[: len(y)]
    return out * (np.std(y) / (np.std(out) + 1e-12))


class Reverberation(Intervention):
    cause = "reverberation"

    def sample_params(self, rng, severity):
        s = jitter_severity(rng, severity)
        return {"rt60_s": lerp(0.15, 1.0, s), "rir_seed": int(rng.integers(0, 2**31 - 1))}

    def _apply(self, y, sr, p, rng):
        rir = synth_rir(sr, p["rt60_s"], np.random.default_rng(p["rir_seed"]))
        return convolve_rir(y, rir)


class Replay(Intervention):
    """Replay attack chain: loudspeaker response, soft nonlinearity, room, microphone noise."""

    cause = "replay"

    def sample_params(self, rng, severity):
        s = jitter_severity(rng, severity)
        return {
            "spk_low_hz": lerp(100, 450, s),
            "spk_high_hz": lerp(9000, 4500, s),
            "drive": lerp(1.0, 3.0, s),
            "rt60_s": lerp(0.1, 0.6, s),
            "mic_snr_db": lerp(40, 22, s),
            "rir_seed": int(rng.integers(0, 2**31 - 1)),
        }

    def _apply(self, y, sr, p, rng):
        hi = min(p["spk_high_hz"], 0.49 * sr)
        sos = signal.butter(4, [p["spk_low_hz"], hi], "band", fs=sr, output="sos")
        x = signal.sosfiltfilt(sos, y)
        peak = np.max(np.abs(x)) + 1e-9
        x = np.tanh(p["drive"] * x / peak) * peak  # loudspeaker non-linearity
        x = convolve_rir(x, synth_rir(sr, p["rt60_s"], np.random.default_rng(p["rir_seed"])))
        x = x + rng.standard_normal(len(x)) * np.std(x) * 10 ** (-p["mic_snr_db"] / 20)
        return x
