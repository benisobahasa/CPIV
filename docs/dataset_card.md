# Dataset card: CPIV benchmark (generated)

**Composition**: for every clean source utterance, a reference copy plus perturbed versions
produced by known intervention chains (identity, single cause, training pairs, held-out
pairs, unknown manipulations). Sampling rate 16 kHz, mono.

**Labels**: multi-label over 8 causes; unknown manipulations carry an all-zero label and
`unknown: true` in the manifest.

**Files**: `manifest.jsonl` (id, source, speaker group, split, cause chain, severity,
parameters), `features.npz`, `causes.json`, optional `audio/`.

**Splits**: speaker-disjoint train/val/test; test is divided into seen, severity, combo and
unknown protocols.

**Licence**: generated audio inherits the licence of the source corpus. Share manifests and
features if redistribution of audio is not allowed.

**Biases**: depends on the source corpus (languages, recording conditions, demographics).
Report these when publishing results.
