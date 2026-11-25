# Changelog

## 0.2.2
- First PyPI release

## 0.2.1
- CI on Linux and macOS, packaging check, CodeQL, release workflow (GitHub Release, optional PyPI), Dependabot, CODEOWNERS, editorconfig, repository setup guide
- `--pattern` and `--group-depth` for real corpora (VCTK, LibriSpeech), `examples/real_pairs_check.py`, getting-started guide

## 0.2.0
- Cause confusion matrix and "most confused causes" in every report
- `cpiv repeat`: multi-seed runs with mean, std and 95% confidence intervals
- `cpiv score`: score any third-party system from a predictions CSV
- `cpiv evaluate-external`: cross-corpus protocol for a trained model
- Z-scores in evidence text are now floored to stay interpretable
- Model card, dataset card, prior-art protocol, security policy, issue/PR templates

## 0.1.0
- Initial release: 12 interventions, 104 features, multi-label attribution with evidence,
  faithfulness tests, benchmark generator and protocols, CLI, CI
