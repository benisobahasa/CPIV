# Benchmark protocol (v0.1)

## Source partitions
Speakers (parent folders) are hashed into train / val / test (about 60/10/30 %). No speaker
appears in more than one partition.

## Conditions per source
- train/val: identity (no change), each known cause alone (severity 0.2-0.65), and the
  training pairs listed in `config.TRAIN_PAIRS`.
- test sources produce four splits:
  - `test_seen`: identity + single causes, severity 0.2-0.65
  - `test_severity`: single causes, severity 0.8-1.0
  - `test_combo`: pairs **not** in `TRAIN_PAIRS`
  - `test_unknown`: manipulations in `config.UNKNOWN_CAUSES`

## Metrics
- Attribution: per-cause F1, macro/micro F1, subset accuracy, Hamming loss, ECE.
- Open-set: AUROC of novelty and of uncertainty (known test splits vs. `test_unknown`),
  plus TPR/FPR at novelty = 1.
- Explanation: deletion and sufficiency gaps at k = 1, 3, 5, 10.
- Confusion: for single-cause samples, mean probability given to every cause (reveals
  overlapping causes such as replay vs. reverberation).

## Reporting checklist
State: corpus, number of speakers, mode (paired/blind), backend, seed, intervention
settings (ffmpeg present? VC proxy or real system?), and that results are single-run unless
repeated over seeds.

## Extending
- Add a held-out corpus directory and call `build_dataset` with the same seed to create a
  cross-corpus test.
- Replace simulated replay with real re-recordings (put them in `test_*` as new conditions).
- Submit a system by producing per-sample cause probabilities for `manifest.jsonl` rows and
  scoring them with `cpiv score` (CSV: `id,<cause names>[,unknown_score]`).
- Use `cpiv repeat` to report mean, standard deviation and 95% confidence intervals over seeds.
- Cross-corpus: build a dataset from another corpus, then `cpiv evaluate-external`.
