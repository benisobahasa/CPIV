"""Unified feature extraction with paired (reference vs test) and blind modes."""

from __future__ import annotations

import numpy as np

from ..config import SAMPLE_RATE
from .phonetic import phonetic_features
from .spectral import spectral_features

# Group -> list of feature-name prefixes/exact names. Used for grouped explanations.
GROUPS: dict[str, list[str]] = {
    "pitch (F0)": ["f0_", "voiced_fraction"],
    "formants": ["f1_", "f2_", "f3_"],
    "voice quality": ["jitter_local", "shimmer_local", "hnr_"],
    "loudness dynamics": ["intensity_", "dyn_range_db", "crest_factor_db", "energy_mod_depth"],
    "spectral shape": ["spec_", "mfcc"],
    "bandwidth / channel": ["e_", "cutoff_hz", "hf_floor_db"],
    "noise / hum": ["noise_floor_db", "hum_prominence_db", "dc_offset"],
    "reverberation": ["decay_slope_db_s"],
    "distortion / quantisation": ["clip_ratio", "amp_level_ratio", "zcr"],
}

# Human readable names for evidence text (phonetician-facing).
DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "f0_mean": ("mean fundamental frequency", "Hz"),
    "f0_median": ("median fundamental frequency", "Hz"),
    "f0_std": ("F0 variability", "Hz"),
    "f0_range": ("F0 range (p95-p5)", "Hz"),
    "voiced_fraction": ("proportion of voiced frames", ""),
    "jitter_local": ("local jitter", ""),
    "shimmer_local": ("local shimmer", ""),
    "hnr_mean": ("harmonics-to-noise ratio", "dB"),
    "hnr_std": ("HNR variability", "dB"),
    "f1_mean": ("first formant (F1)", "Hz"),
    "f2_mean": ("second formant (F2)", "Hz"),
    "f3_mean": ("third formant (F3)", "Hz"),
    "f1_bw": ("F1 bandwidth", "Hz"),
    "f2_bw": ("F2 bandwidth", "Hz"),
    "intensity_mean": ("mean intensity", "dB"),
    "intensity_std": ("intensity variability", "dB"),
    "spec_centroid": ("spectral centroid", "Hz"),
    "spec_bandwidth": ("spectral bandwidth", "Hz"),
    "spec_rolloff95": ("95% spectral roll-off", "Hz"),
    "spec_flatness": ("spectral flatness", ""),
    "spec_flux": ("spectral flux", ""),
    "spec_tilt": ("spectral tilt", "dB/oct"),
    "zcr": ("zero-crossing rate", ""),
    "e_0_200": ("energy share 0-200 Hz", ""),
    "e_200_300": ("energy share 200-300 Hz", ""),
    "e_300_3400": ("energy share 300-3400 Hz (telephone band)", ""),
    "e_3400_4000": ("energy share 3.4-4 kHz", ""),
    "e_4k_8k": ("energy share 4-8 kHz", ""),
    "cutoff_hz_40db": ("effective bandwidth (-40 dB)", "Hz"),
    "cutoff_hz_60db": ("effective bandwidth (-60 dB)", "Hz"),
    "hf_floor_db": ("high-frequency floor relative to peak", "dB"),
    "hum_prominence_db": ("mains-hum prominence", "dB"),
    "noise_floor_db": ("noise floor", "dB"),
    "dyn_range_db": ("frame-energy dynamic range", "dB"),
    "decay_slope_db_s": ("steepest energy decay rate", "dB/s"),
    "energy_mod_depth": ("energy modulation depth", ""),
    "clip_ratio": ("clipped-sample ratio", ""),
    "crest_factor_db": ("crest factor", "dB"),
    "dc_offset": ("DC offset", ""),
    "amp_level_ratio": ("distinct amplitude levels ratio", ""),
}
for _i in range(1, 13):
    DESCRIPTIONS[f"mfcc{_i}"] = (f"MFCC {_i}", "")


def group_of(feature: str) -> str:
    base = feature.split("::", 1)[-1]
    for g, keys in GROUPS.items():
        if any(base == k or base.startswith(k) for k in keys):
            return g
    return "other"


def describe(feature: str) -> tuple[str, str]:
    base = feature.split("::", 1)[-1]
    return DESCRIPTIONS.get(base, (base, ""))


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """All features of a single signal."""
    y = np.asarray(y, dtype=np.float64)
    feats = {}
    feats.update(phonetic_features(y, sr))
    feats.update(spectral_features(y.astype(np.float32), sr))
    return {k: float(v) for k, v in feats.items()}


def feature_vector(test: np.ndarray, reference: np.ndarray | None = None, sr: int = SAMPLE_RATE,
                   ) -> tuple[np.ndarray, list[str]]:
    """Build the model input.

    * paired mode (``reference`` given): ``[delta::name ..., test::name ...]``
      where delta = test - reference. This is what makes the task "what *changed*?".
    * blind mode: only ``test::name`` features.
    """
    ft = extract_features(test, sr)
    names = sorted(ft)
    if reference is None:
        return np.array([ft[n] for n in names]), [f"test::{n}" for n in names]
    fr = extract_features(reference, sr)
    delta = [ft[n] - fr[n] for n in names]
    return (np.array(delta + [ft[n] for n in names]),
            [f"delta::{n}" for n in names] + [f"test::{n}" for n in names])
