# Model card: CPIV attribution model (v0.2)

**Task**: multi-label attribution of the causes behind the difference between a reference
and a test recording, with evidence and an unknown-manipulation novelty score.

**Model**: one-vs-rest random forest (default) over 104 interpretable features;
Ledoit-Wolf Mahalanobis novelty in standardised feature space.

**Training data**: whatever clean corpus you supply, perturbed by the controlled
interventions in this repository. No pre-trained weights are distributed.

**Intended use**: research, benchmarking, defensive triage with expert review.

**Out of scope**: legal evidence, health or emotion inference, identifying individuals,
detecting manipulations by systems far outside the simulated families without further validation.

**Evaluation**: protocols in [benchmark_protocol.md](benchmark_protocol.md); report
seeds, confidence intervals, mode and corpus.

**Known limitations**: simulated replay/channel, proxy voice conversion, correlated features,
explanations reflect model reliance rather than physical proof, results on synthetic voices
are not meaningful for real speech.
