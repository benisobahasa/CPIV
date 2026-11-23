"""Phonetically interpretable features computed with Praat (via Parselmouth)."""

from __future__ import annotations

import numpy as np

NAN_FILL = 0.0


def _nz(x: float) -> float:
    return NAN_FILL if x is None or not np.isfinite(x) else float(x)


def phonetic_features(y: np.ndarray, sr: int) -> dict[str, float]:
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    out: dict[str, float] = {}

    # --- F0 ---
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    out["voiced_fraction"] = float(len(voiced) / max(len(f0), 1))
    if len(voiced) > 3:
        out["f0_mean"] = float(np.mean(voiced))
        out["f0_median"] = float(np.median(voiced))
        out["f0_std"] = float(np.std(voiced))
        out["f0_range"] = float(np.percentile(voiced, 95) - np.percentile(voiced, 5))
    else:
        for k in ("f0_mean", "f0_median", "f0_std", "f0_range"):
            out[k] = NAN_FILL

    # --- voice quality: jitter, shimmer, HNR ---
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 60, 500)
        out["jitter_local"] = _nz(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        out["shimmer_local"] = _nz(call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception:
        out["jitter_local"] = out["shimmer_local"] = NAN_FILL
    try:
        harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=60)
        h = harm.values[harm.values > -200]
        out["hnr_mean"] = float(np.mean(h)) if len(h) else NAN_FILL
        out["hnr_std"] = float(np.std(h)) if len(h) else NAN_FILL
    except Exception:
        out["hnr_mean"] = out["hnr_std"] = NAN_FILL

    # --- formants ---
    try:
        fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500,
                                 window_length=0.025, pre_emphasis_from=50)
        ts = fm.xs()
        for i in (1, 2, 3):
            vals = np.array([fm.get_value_at_time(i, t) for t in ts])
            vals = vals[np.isfinite(vals)]
            out[f"f{i}_mean"] = float(np.median(vals)) if len(vals) else NAN_FILL
        for i in (1, 2):
            vals = np.array([fm.get_bandwidth_at_time(i, t) for t in ts])
            vals = vals[np.isfinite(vals)]
            out[f"f{i}_bw"] = float(np.median(vals)) if len(vals) else NAN_FILL
    except Exception:
        for k in ("f1_mean", "f2_mean", "f3_mean", "f1_bw", "f2_bw"):
            out[k] = NAN_FILL

    # --- intensity dynamics ---
    inten = snd.to_intensity(minimum_pitch=100, time_step=0.01).values.flatten()
    inten = inten[np.isfinite(inten)]
    out["intensity_mean"] = float(np.mean(inten)) if len(inten) else NAN_FILL
    out["intensity_std"] = float(np.std(inten)) if len(inten) else NAN_FILL
    return out
