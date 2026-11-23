# Methodology

## Problem
Given a reference recording `r` and a test recording `t` of the same content, estimate the
set `C` of causes responsible for the difference, and justify each cause with measurable
evidence. This differs from binary spoof detection (genuine vs. fake) and from source
tracing (which generator made a fake).

## Causal design
Ground truth comes from **interventions**: start from a clean `r`, apply a chosen chain of
treatments `do(c1, c2, ...)` with sampled parameters, obtain `t`. Because `t` is produced
only by those treatments, the label is certain. Parameters and severities are logged in
the manifest.

Known causes: telephone channel, codec compression, replay chain, additive noise,
reverberation, voice-conversion proxy, pitch manipulation, natural variation.

Overlap is real: replay contains reverberation and noise. The benchmark keeps the labels
as defined by the *treatment applied* and reports confusion explicitly instead of hiding it.

## Features
- Praat/Parselmouth: F0 statistics, jitter, shimmer, HNR, F1-F3 and bandwidths, intensity.
- librosa/SciPy: spectral shape, MFCC means, band energy shares, effective bandwidth,
  spectral tilt, mains-hum prominence, noise floor, energy decay slope, clipping,
  quantisation cue.
- Paired mode input is `[test - reference, test]`; blind mode uses only `test`.

## Model
One-vs-rest classifier per cause (random forest by default; `hgb` and `logreg` available).
Multi-label output allows combinations. Open-set: Ledoit-Wolf Mahalanobis distance in
standardised feature space, normalised so that 1.0 is the 99th percentile of known data.

## Evidence
For each feature `j` and cause `c`: occlusion importance
`P(c | x) - P(c | x with feature j set to its no-change baseline)`; the baseline is the
median over training rows with no intervention. Group-level importance occludes whole
phonetic groups (pitch, formants, voice quality, ...). Each evidence item carries a plain
description, unit, observed value and a robust z-score against the no-change distribution.

## Faithfulness
- Deletion gap: drop in P(c) when removing the top-k evidence features minus the drop when
  removing k random features.
- Sufficiency gap: retained P(c) when keeping only the top-k evidence features minus the
  same for k random features.
Positive gaps indicate explanations are not decorative. Occlusion is not causal proof of a
physical mechanism; it measures model reliance.

## Threats to validity
Simulated channels and proxy VC; feature-level (not waveform-level) models; correlated
features can split importance; corpus effects. All must be addressed with real data before
strong claims.
