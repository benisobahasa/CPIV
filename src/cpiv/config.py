"""Global constants shared across CPIV."""

from __future__ import annotations

SAMPLE_RATE = 16000

# Known causes a model is trained to attribute. Order defines label-vector order.
CAUSES: list[str] = [
    "telephone_channel",   # narrowband band-pass + G.711-style companding
    "codec_compression",   # lossy codec round trip (mp3/opus or built-in simulation)
    "replay",              # loudspeaker -> room -> microphone chain
    "additive_noise",      # colored noise / mains hum at a given SNR
    "reverberation",       # room impulse response only
    "voice_conversion",    # lightweight proxy (pitch + formant shift); plug in real VC systems
    "pitch_manipulation",  # explicit F0 shift
    "natural_variation",   # vocal effort / speaking rate / level, NOT a medical claim
]

# Manipulations that are never used for training. Used to test open-set behaviour.
UNKNOWN_CAUSES: list[str] = [
    "unk_bitcrush",
    "unk_spectral_hole",
    "unk_time_warp",
    "unk_phase_noise",
]

# Combination pairs that appear during training (the rest are held out for test_combo).
TRAIN_PAIRS: list[tuple[str, str]] = [
    ("telephone_channel", "additive_noise"),
    ("replay", "additive_noise"),
    ("codec_compression", "reverberation"),
    ("pitch_manipulation", "natural_variation"),
]

SPLITS = ["train", "val", "test_seen", "test_severity", "test_combo", "test_unknown"]
