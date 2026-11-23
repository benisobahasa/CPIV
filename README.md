# CPIV: Causal Perturbation & Intervention for Voice

CPIV is an open-source toolkit and benchmark for answering a question that most voice
forensics systems skip: **why did this voice recording change?**

Instead of a single "genuine / fake" score, CPIV takes a reference recording and a test
recording and estimates which known causes explain the difference (telephone channel,
codec compression, replay, noise, reverberation, voice conversion, pitch manipulation,
natural speaker variation). Every attribution comes with **evidence**: phonetically
interpretable measurements (F0, formants, jitter, shimmer, HNR, spectral tilt, bandwidth,
decay rate, ...) and a **faithfulness test** showing those measurements really drive the
decision.

> **Status: research prototype (v0.2).** The pipeline, benchmark generator and
> explanation tests are complete and tested. Scientific claims still require
> experiments on real corpora (see [docs/roadmap.md](docs/roadmap.md)). CPIV makes no
> claim to be a finished or competitor-free technology.

## Why this design

| Idea | How CPIV implements it |
|---|---|
| Causal ground truth | Each test clip is produced by **controlled interventions** on a clean recording, so the true cause is known by construction |
| Multi-cause reality | Labels are multi-label; causes can be combined (`replay + additive_noise`) |
| Evidence, not scores | Occlusion-based evidence per cause with phonetic descriptions, z-scores and group-level contributions |
| Trustworthy explanations | Deletion and sufficiency tests compare evidence features against random features |
| Unknown manipulations | Four manipulations are never trained on; a novelty score flags patterns outside the known cause space |
| Robustness protocols | Severity shift, held-out cause combinations, unseen manipulations, optional cross-corpus data |

## Install

```bash
git clone https://github.com/benisobahasa/CPIV.git
cd CPIV
pip install -e ".[dev]"
# optional: ffmpeg on PATH enables real mp3/opus round trips for the codec intervention
```

Requires Python 3.10+. No GPU, paid API or server needed.

> New here? Follow [docs/getting_started.md](docs/getting_started.md) for a macOS walk-through
> from install to a real-corpus experiment.

## 60-second demo (synthetic voices, no downloads)

```bash
cpiv demo --out runs/demo
```

This builds a small benchmark, trains the model, prints the protocol report and prints a
text explanation for an example "replay + noise" recording. Synthetic voices are only for
smoke tests; they are not speech.

## Use on real data

```bash
# corpus layout: <root>/<speaker_id>/<utterance>.wav  (one folder per speaker)
cpiv build-benchmark --sources /path/to/corpus --max-files 600 --out runs/real/data
cpiv benchmark --data runs/real/data --out runs/real/paired --mode paired
cpiv benchmark --data runs/real/data --out runs/real/blind  --mode blind

# repeat over seeds with 95% confidence intervals
cpiv repeat --sources /path/to/corpus --out runs/real/repeat --seeds 0 1 2 3 4 --modes paired blind

# cross-corpus test: build a dataset from another corpus, evaluate the trained model on it
cpiv evaluate-external --model runs/real/paired/model.joblib --data runs/other/data

# score any third-party system from a predictions CSV
cpiv score --data runs/real/data --predictions preds.csv

# explain a new pair with the trained model
cpiv attribute --model runs/real/paired/model.joblib --test test.wav --reference ref.wav
```

Splits are speaker-disjoint (the parent folder name is the speaker id).

### Python API

```python
from cpiv.api import attribute_pair
result = attribute_pair("runs/real/paired/model.joblib", "test.wav", "ref.wav")
print(result.text())          # human-readable evidence
print(result.to_dict())       # machine-readable
```

See [examples/quickstart.py](examples/quickstart.py).

## Benchmark protocols

| Split | Question |
|---|---|
| `test_seen` | Known causes, known severity range, unseen speakers |
| `test_severity` | Same causes at stronger severity than training |
| `test_combo` | Cause **combinations never seen** in training |
| `test_unknown` | Manipulations outside the cause list (bit-crush, spectral hole, time warp, phase noise) |

Metrics: macro/micro F1, subset accuracy, Hamming loss, calibration error (ECE), open-set
AUROC (novelty and uncertainty), faithfulness gaps, cause-confusion matrix. Details in
[docs/benchmark_protocol.md](docs/benchmark_protocol.md).

Every build writes `manifest.jsonl` (cause chain, parameters, split) next to the audio, so
other systems can be evaluated on exactly the same files.

## Repository layout

```
src/cpiv/
  interventions/   12 controlled treatments (8 known causes + 4 unknown manipulations)
  features/        Praat/Parselmouth phonetic features + spectral/channel cues
  attribution/     multi-label model, open-set novelty, evidence, faithfulness tests
  benchmark/       source splitting, dataset generation, metrics, protocol runner
  cli.py           `cpiv` command
tests/             pytest suite (CI on Python 3.10-3.12)
docs/              methodology, protocol, CLI, model/dataset cards, prior-art protocol, roadmap
```

## Known limitations (please read)

- **Voice conversion is a proxy** (joint F0 + formant shift). Plug real VC/voice-cloning
  systems in via `ExternalVoiceConversion` ([example](examples/plug_in_voice_conversion.py)).
- **Replay and reverberation are simulated** with a statistical room model; real
  re-recordings should be added to the benchmark before publishing results.
- **Causes overlap physically** (replay includes room and noise); this is intentional and
  documented in the methodology.
- Results on the built-in synthetic voices say nothing about real speech.
- Explanations describe what the model relies on, not proof of a physical mechanism.

## Responsible use

CPIV is meant for research and defensive forensics. Read [ETHICS.md](ETHICS.md):
dual-use risk, no health or emotion inference, and not for use as sole evidence in legal
proceedings.

## Citing

See [CITATION.cff](CITATION.cff). License: MIT.
