# Roadmap

- [x] v0.1: interventions, features, attribution + evidence, faithfulness, benchmark, CLI, CI
- [x] v0.2: confusion reporting, multi-seed CIs, external scoring, cross-corpus evaluation, model/dataset cards
- [ ] Prior-art review (papers, patents, datasets): spoofing, source tracing, forensic
      phonetics, causal audio analysis. Record findings in `docs/prior_art.md`.
- [ ] Real corpora experiments (VCTK/LibriSpeech) using `cpiv repeat` (tooling ready, runs pending)
- [ ] Real re-recorded and telephone evaluation sets
- [ ] Neural VC / voice-cloning adapters for the `voice_conversion` cause
- [ ] Waveform-level / SSL-embedding baselines for comparison with the feature model
- [ ] Better replay simulation (measured impulse responses)
- [ ] Likelihood-ratio style calibration for forensic reporting
- [ ] Collaboration with signal-processing / digital-forensics experts
