"""Audio helpers: loading, normalisation, and a tiny formant synthesiser for demos/tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .config import SAMPLE_RATE


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as mono float32 at ``sr``."""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def save_audio(path: str | Path, y: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(y, -1.0, 1.0).astype(np.float32), sr, subtype="PCM_16")


def peak_normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return y if m < 1e-9 else (y * (peak / m)).astype(np.float32)


def rms_db(y: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(y))) + 1e-12))


def match_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def synth_voice(
    seed: int = 0,
    duration: float = 3.0,
    sr: int = SAMPLE_RATE,
    f0_base: float | None = None,
    formant_scale: float | None = None,
) -> np.ndarray:
    """Generate a synthetic vowel-like utterance (source-filter model).

    This is NOT a substitute for real speech. It exists so the repository, tests and
    demo run with no downloads. Use real corpora for any scientific claim.
    """
    rng = np.random.default_rng(seed)
    f0_base = f0_base if f0_base is not None else rng.uniform(90, 240)
    formant_scale = formant_scale if formant_scale is not None else rng.uniform(0.85, 1.2)
    n = int(duration * sr)
    t = np.arange(n) / sr

    # F0 contour: slow declination + vibrato-like modulation
    f0 = f0_base * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.6, 1.4) * t)
                    + 0.03 * np.sin(2 * np.pi * 4.5 * t)) * (1 - 0.1 * t / duration)

    # glottal pulse train with small period jitter
    phase = np.cumsum(f0 / sr)
    pulses = np.zeros(n)
    idx = np.flatnonzero(np.diff(np.floor(phase)) > 0)
    pulses[idx] = 1.0
    # shape pulses with a Rosenberg-like smoothing and shimmer
    pulses *= 1 + 0.02 * rng.standard_normal(n)
    b, a = signal.butter(2, 1000 / (sr / 2), "low")
    source = signal.lfilter(b, a, pulses)
    source += 0.01 * rng.standard_normal(n)  # aspiration noise

    # vowel sequence with moving formants
    vowels = np.array([[730, 1090, 2440], [270, 2290, 3010], [300, 870, 2240],
                       [530, 1840, 2480], [660, 1720, 2410]], dtype=float)
    seq = vowels[rng.integers(0, len(vowels), size=5)] * formant_scale
    seg = n // len(seq)
    out = np.zeros(n)
    for i, fm in enumerate(seq):
        s, e = i * seg, (i + 1) * seg if i < len(seq) - 1 else n
        x = source[s:e]
        for freq, bw in zip(fm, (80, 110, 160), strict=True):
            freq = min(freq, 0.45 * sr)
            r = np.exp(-np.pi * bw / sr)
            th = 2 * np.pi * freq / sr
            x = signal.lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)
        out[s:e] = x
    # syllabic amplitude envelope
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, 6.28)) ** 2
    fade = np.minimum(1, np.minimum(t, duration - t) / 0.05)
    out = out * env * fade
    return peak_normalize(out.astype(np.float32), 0.8)
