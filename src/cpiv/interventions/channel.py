"""Telephone channel, codec compression and additive noise interventions."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .base import Intervention, jitter_severity, lerp


def _bandpass(y, sr, lo, hi, order=6):
    hi = min(hi, 0.49 * sr)
    sos = signal.butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y)


def _lowpass(y, sr, hi, order=8):
    sos = signal.butter(order, min(hi, 0.49 * sr), btype="low", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y)


def mu_law_roundtrip(y: np.ndarray, mu: int = 255, bits: int = 8) -> np.ndarray:
    y = np.clip(y, -1, 1)
    comp = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    levels = 2 ** bits
    q = np.round((comp + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1
    return np.sign(q) * (1 / mu) * ((1 + mu) ** np.abs(q) - 1)


def a_law_roundtrip(y: np.ndarray, A: float = 87.6, bits: int = 8) -> np.ndarray:
    y = np.clip(y, -1, 1)
    ay = np.abs(y)
    comp = np.where(ay < 1 / A, A * ay / (1 + np.log(A)),
                    (1 + np.log(np.maximum(A * ay, 1e-12))) / (1 + np.log(A)))
    comp = np.sign(y) * comp
    levels = 2 ** bits
    q = np.round((comp + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1
    aq = np.abs(q)
    out = np.where(aq < 1 / (1 + np.log(A)), aq * (1 + np.log(A)) / A,
                   np.exp(aq * (1 + np.log(A)) - 1) / A)
    return np.sign(q) * out


class TelephoneChannel(Intervention):
    """Narrowband telephony: band-pass, 8 kHz resampling, companding, mild line noise."""

    cause = "telephone_channel"

    def sample_params(self, rng, severity):
        s = jitter_severity(rng, severity)
        return {
            "low_hz": lerp(200, 400, s),
            "high_hz": lerp(3800, 3000, s),
            "law": str(rng.choice(["mu", "a"])),
            "bits": int(round(lerp(10, 7, s))),
            "line_snr_db": lerp(45, 28, s),
        }

    def _apply(self, y, sr, p, rng):
        x = _bandpass(y, sr, p["low_hz"], p["high_hz"])
        x = signal.resample_poly(x, 8000, sr)
        peak = np.max(np.abs(x)) + 1e-9
        x = x / peak
        x = mu_law_roundtrip(x, bits=p["bits"]) if p["law"] == "mu" else a_law_roundtrip(x, bits=p["bits"])
        x = x * peak
        x = x + rng.standard_normal(len(x)) * np.std(x) * 10 ** (-p["line_snr_db"] / 20)
        return signal.resample_poly(x, sr, 8000)


class CodecCompression(Intervention):
    """Lossy codec round trip. Uses ffmpeg (mp3/opus) when present, else a built-in
    STFT-quantisation simulation so the pipeline works everywhere."""

    cause = "codec_compression"

    def __init__(self, use_ffmpeg: bool = True):
        self.use_ffmpeg = use_ffmpeg and shutil.which("ffmpeg") is not None

    def sample_params(self, rng, severity):
        s = jitter_severity(rng, severity)
        codec = str(rng.choice(["mp3", "opus"])) if self.use_ffmpeg else "sim"
        return {"codec": codec, "bitrate_kbps": int(round(lerp(96, 12, s))),
                "cutoff_hz": lerp(7500, 3500, s), "q_step_db": lerp(1.5, 9.0, s)}

    def _ffmpeg(self, y, sr, codec, kbps):
        with tempfile.TemporaryDirectory() as d:
            src, enc, dec = Path(d) / "a.wav", Path(d) / f"a.{'mp3' if codec == 'mp3' else 'ogg'}", Path(d) / "b.wav"
            sf.write(str(src), y.astype(np.float32), sr)
            lib = ["-c:a", "libmp3lame"] if codec == "mp3" else ["-c:a", "libopus"]
            kbps = max(kbps, 8) if codec == "mp3" else max(kbps, 6)
            r1 = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *lib,
                                 "-b:a", f"{kbps}k", str(enc)], capture_output=True)
            if r1.returncode != 0:
                raise RuntimeError(r1.stderr.decode()[:200])
            r2 = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(enc),
                                 "-ar", str(sr), "-ac", "1", str(dec)], capture_output=True)
            if r2.returncode != 0:
                raise RuntimeError(r2.stderr.decode()[:200])
            out, _ = sf.read(str(dec))
        return out

    def _sim(self, y, sr, p):
        f, t, Z = signal.stft(y, fs=sr, nperseg=512, noverlap=384)
        mag, ph = np.abs(Z), np.angle(Z)
        logm = 20 * np.log10(mag + 1e-8)
        step = p["q_step_db"]
        logm = np.round(logm / step) * step
        mag = 10 ** (logm / 20)
        mag[f > p["cutoff_hz"], :] = 0
        mag[mag < 10 ** (-70 / 20)] = 0
        _, x = signal.istft(mag * np.exp(1j * ph), fs=sr, nperseg=512, noverlap=384)
        return x

    def _apply(self, y, sr, p, rng):
        if p["codec"] in ("mp3", "opus") and self.use_ffmpeg:
            try:
                return self._ffmpeg(y, sr, p["codec"], p["bitrate_kbps"])
            except Exception:  # fall back gracefully
                p["codec"] = "sim"
        return self._sim(y, sr, p)


def _colored_noise(n: int, color: str, rng) -> np.ndarray:
    w = rng.standard_normal(n)
    if color == "white":
        return w
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    exp = {"pink": 0.5, "brown": 1.0}[color]
    return np.fft.irfft(spec / f ** exp, n)


class AdditiveNoise(Intervention):
    cause = "additive_noise"

    def sample_params(self, rng, severity):
        s = jitter_severity(rng, severity)
        return {"snr_db": lerp(30, 3, s), "color": str(rng.choice(["white", "pink", "brown"])),
                "hum_hz": float(rng.choice([0, 50, 60])), "hum_db_rel": lerp(-35, -15, s)}

    def _apply(self, y, sr, p, rng):
        n = len(y)
        noise = _colored_noise(n, p["color"], rng)
        noise = noise / (np.std(noise) + 1e-12)
        sig_p = np.mean(y ** 2) + 1e-12
        noise *= np.sqrt(sig_p / 10 ** (p["snr_db"] / 10))
        out = y + noise
        if p["hum_hz"] > 0:
            t = np.arange(n) / sr
            hum = sum(np.sin(2 * np.pi * p["hum_hz"] * k * t) / k for k in (1, 2, 3))
            out = out + hum * np.sqrt(sig_p) * 10 ** (p["hum_db_rel"] / 20)
        return out
