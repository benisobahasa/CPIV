"""Spectral-shape, channel and acoustic-environment cues (librosa / NumPy / SciPy)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def _ltas(y: np.ndarray, sr: int, nperseg: int = 1024):
    f, p = signal.welch(y, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    return f, 10 * np.log10(p + 1e-14)


def _band_ratio(f, p_lin, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.sum(p_lin[m]) / (np.sum(p_lin) + 1e-14))


def spectral_features(y: np.ndarray, sr: int) -> dict[str, float]:
    import librosa

    out: dict[str, float] = {}
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=160)) + 1e-10
    out["spec_centroid"] = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    out["spec_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    out["spec_rolloff95"] = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)))
    out["spec_flatness"] = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    out["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=160)))
    flux = np.sqrt(np.sum(np.diff(S, axis=1).clip(min=0) ** 2, axis=0))
    out["spec_flux"] = float(np.mean(flux) / (np.mean(S) + 1e-9))

    mf = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=sr)), n_mfcc=13)
    for i in range(1, 13):
        out[f"mfcc{i}"] = float(np.mean(mf[i]))

    # long-term average spectrum features
    f, p_db = _ltas(y, sr)
    p_lin = 10 ** (p_db / 10)
    m = (f >= 100) & (f <= 4000)
    out["spec_tilt"] = float(np.polyfit(np.log2(f[m]), p_db[m], 1)[0])  # dB / octave
    for name, (lo, hi) in {"e_0_200": (0, 200), "e_200_300": (200, 300), "e_300_3400": (300, 3400),
                           "e_3400_4000": (3400, 4000), "e_4k_8k": (4000, 8000)}.items():
        out[name] = _band_ratio(f, p_lin, lo, hi)
    peak = float(np.max(p_db))
    above = np.flatnonzero(p_db > peak - 40)
    out["cutoff_hz_40db"] = float(f[above[-1]]) if len(above) else 0.0
    out["cutoff_hz_60db"] = float(f[np.flatnonzero(p_db > peak - 60)[-1]])
    out["hf_floor_db"] = float(np.mean(p_db[(f > 6000) & (f < 7800)]) - peak)

    # hum (mains) cue
    hum = 0.0
    for h in (50, 60):
        band = (f > h - 3) & (f < h + 3)
        side = ((f > h - 15) & (f < h - 6)) | ((f > h + 6) & (f < h + 15))
        if band.any() and side.any():
            hum = max(hum, float(np.max(p_db[band]) - np.mean(p_db[side])))
    out["hum_prominence_db"] = hum

    # frame-energy dynamics (noise floor, SNR proxy, decay for reverberation)
    fr = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    e_db = 20 * np.log10(fr + 1e-9)
    out["noise_floor_db"] = float(np.percentile(e_db, 10))
    out["dyn_range_db"] = float(np.percentile(e_db, 95) - np.percentile(e_db, 10))
    d = np.diff(e_db) / 0.01
    neg = d[d < 0]
    out["decay_slope_db_s"] = float(np.percentile(neg, 10)) if len(neg) > 5 else 0.0
    out["energy_mod_depth"] = float(np.std(fr) / (np.mean(fr) + 1e-9))

    # waveform level cues
    out["clip_ratio"] = float(np.mean(np.abs(y) > 0.98 * (np.max(np.abs(y)) + 1e-9)))
    out["crest_factor_db"] = float(20 * np.log10((np.max(np.abs(y)) + 1e-9) / (np.sqrt(np.mean(y ** 2)) + 1e-9)))
    out["dc_offset"] = float(abs(np.mean(y)))
    # quantisation cue: fraction of distinct amplitude levels (low for bit-crushed audio)
    out["amp_level_ratio"] = float(len(np.unique(np.round(y * 2 ** 15))) / max(len(y), 1))
    return out
