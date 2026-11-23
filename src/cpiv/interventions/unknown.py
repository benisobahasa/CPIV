"""Manipulations never seen during training; used for open-set / unknown-cause tests."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .base import Intervention, jitter_severity, lerp


class BitCrush(Intervention):
    cause = "unk_bitcrush"

    def sample_params(self, rng, severity):
        s = jitter_severity(rng, severity)
        return {"bits": int(round(lerp(10, 4, s))), "downsample": int(round(lerp(1, 4, s)))}

    def _apply(self, y, sr, p, rng):
        levels = 2 ** (p["bits"] - 1)
        x = np.round(y / (np.max(np.abs(y)) + 1e-9) * levels) / levels
        if p["downsample"] > 1:
            x = np.repeat(x[:: p["downsample"]], p["downsample"])[: len(y)]
        return x


class SpectralHole(Intervention):
    cause = "unk_spectral_hole"

    def sample_params(self, rng, severity):
        s = jitter_severity(rng, severity)
        c = float(rng.uniform(800, 5000))
        return {"center_hz": c, "width_hz": lerp(150, 900, s), "depth_db": lerp(15, 45, s)}

    def _apply(self, y, sr, p, rng):
        f = np.fft.rfftfreq(len(y), 1 / sr)
        g = 1 - (1 - 10 ** (-p["depth_db"] / 20)) * np.exp(-0.5 * ((f - p["center_hz"]) / (p["width_hz"] / 2)) ** 2)
        return np.fft.irfft(np.fft.rfft(y) * g, len(y))


class TimeWarp(Intervention):
    cause = "unk_time_warp"

    def sample_params(self, rng, severity):
        s = jitter_severity(rng, severity)
        return {"depth": lerp(0.005, 0.04, s), "rate_hz": float(rng.uniform(0.5, 3.0))}

    def _apply(self, y, sr, p, rng):
        n = len(y)
        t = np.arange(n)
        warp = t + p["depth"] * sr * np.sin(2 * np.pi * p["rate_hz"] * t / sr) / (2 * np.pi * p["rate_hz"])
        return np.interp(warp, t, y)


class PhaseNoise(Intervention):
    cause = "unk_phase_noise"

    def sample_params(self, rng, severity):
        s = jitter_severity(rng, severity)
        return {"phase_std": lerp(0.1, 1.2, s)}

    def _apply(self, y, sr, p, rng):
        f, t, Z = signal.stft(y, fs=sr, nperseg=512, noverlap=384)
        Z = np.abs(Z) * np.exp(1j * (np.angle(Z) + rng.normal(0, p["phase_std"], Z.shape)))
        return signal.istft(Z, fs=sr, nperseg=512, noverlap=384)[1]
