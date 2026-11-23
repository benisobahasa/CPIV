"""Speaker-side interventions: pitch, voice-conversion proxy and natural variation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from .base import Intervention, jitter_severity, lerp


class PitchManipulation(Intervention):
    cause = "pitch_manipulation"

    def sample_params(self, rng, severity):
        s = jitter_severity(rng, severity)
        sign = float(rng.choice([-1, 1]))
        return {"semitones": sign * lerp(1.0, 6.0, s)}

    def _apply(self, y, sr, p, rng):
        import librosa

        return librosa.effects.pitch_shift(y, sr=sr, n_steps=p["semitones"])


class VoiceConversionProxy(Intervention):
    """Lightweight stand-in for voice conversion: joint F0 + formant (vocal tract) change.

    Real neural VC / voice cloning systems should be plugged in through
    :class:`ExternalVoiceConversion`. This proxy only mimics the *kind* of change
    (speaker identity features move together) and must be described as a proxy in papers.
    """

    cause = "voice_conversion"

    def sample_params(self, rng, severity):
        s = jitter_severity(rng, severity)
        sign = float(rng.choice([-1, 1]))
        return {"formant_ratio": float(1 + sign * lerp(0.05, 0.25, s)),
                "pitch_ratio": float(1 + sign * lerp(0.1, 0.6, s)),
                "pitch_range": float(rng.uniform(0.8, 1.2))}

    def _apply(self, y, sr, p, rng):
        try:
            import parselmouth
            from parselmouth.praat import call

            snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
            out = call(snd, "Change gender", 75, 600, p["formant_ratio"], 0,
                       p["pitch_ratio"], p["pitch_range"])
            return np.asarray(out.values[0])
        except Exception:
            import librosa

            n_steps = 12 * np.log2(p["pitch_ratio"])
            x = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            r = p["formant_ratio"]
            return signal.resample(signal.resample(x, int(len(x) / r)), len(x))


class ExternalVoiceConversion(Intervention):
    """Adapter for a user supplied voice-conversion callable ``fn(y, sr) -> y2``."""

    cause = "voice_conversion"

    def __init__(self, fn, name: str = "external"):
        self.fn, self.name = fn, name

    def sample_params(self, rng, severity):
        return {"system": self.name}

    def _apply(self, y, sr, p: dict[str, Any], rng):
        return np.asarray(self.fn(y.astype(np.float32), sr), dtype=np.float64)


class NaturalVariation(Intervention):
    """Everyday speaker variation: vocal effort (spectral tilt), speaking rate, level.

    This is intentionally NOT linked to health or emotion inference.
    """

    cause = "natural_variation"

    def sample_params(self, rng, severity):
        s = jitter_severity(rng, severity)
        sign = float(rng.choice([-1, 1]))
        return {"tilt_db_per_oct": sign * lerp(1.0, 6.0, s),
                "rate": float(1 + rng.choice([-1, 1]) * lerp(0.03, 0.2, s)),
                "gain_db": float(rng.uniform(-6, 6))}

    def _apply(self, y, sr, p, rng):
        import librosa

        n = len(y)
        spec = np.fft.rfft(y)
        f = np.fft.rfftfreq(n, 1 / sr)
        oct_ = np.log2(np.maximum(f, 100) / 1000.0)
        gain = 10 ** ((p["tilt_db_per_oct"] * oct_) / 20)
        x = np.fft.irfft(spec * gain, n)
        x = librosa.effects.time_stretch(x, rate=p["rate"])
        return x * 10 ** (p["gain_db"] / 20)
