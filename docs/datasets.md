# Datasets

CPIV generates its own perturbations; you only need **clean** speech with speaker ids.
Candidates (verify each licence and terms before use):

- VCTK, LibriSpeech, Common Voice: clean read speech; good starting points.
- ASVspoof series, CodecFake-style and MLAAD-style datasets: for external evaluation of
  spoofing, codec and generator conditions (not required to run CPIV).
- Real telephone and re-recorded speech: strongly recommended as an evaluation-only set.

Layout expected by `cpiv build-benchmark --sources`:

```
corpus/
  speaker_001/utt1.wav
  speaker_001/utt2.wav
  speaker_002/utt1.wav
```

Do not redistribute derived audio if the source licence forbids it; share the manifest and
feature files instead.

## Verified download pages
- CSTR VCTK Corpus 0.92 (CC BY 4.0): https://datashare.ed.ac.uk/handle/10283/3443
- Device Recorded VCTK: https://datashare.ed.ac.uk/handle/10283/3038
- ASVspoof 2017 v2 (real replay): https://datashare.ed.ac.uk/handle/10283/3055
- ASVspoof 2019 (LA, PA): https://datashare.ed.ac.uk/handle/10283/3336
- ReMASC paper: https://arxiv.org/abs/1904.03365

LibriSpeech (https://www.openslr.org/12) and Mozilla Common Voice
(https://commonvoice.mozilla.org/datasets) are well-known open corpora; confirm their current
licences on those pages before use.

Step-by-step usage: [getting_started.md](getting_started.md).
