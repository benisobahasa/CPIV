# Getting started (macOS)

## 1. Install tools (once)
```bash
# Homebrew: https://brew.sh (skip if `brew --version` works)
brew install python@3.12 ffmpeg libsndfile git
```

## 2. Get the code
Before the first push to GitHub, use the zip you were given:
```bash
mkdir -p ~/Projects && unzip ~/Downloads/CPIV.zip -d ~/Projects
cd ~/Projects/CPIV
```
After the repository is public, anyone can instead run:
```bash
git clone https://github.com/benisobahasa/CPIV.git && cd CPIV
```

## 3. Install and verify
```bash
python3.12 -m venv .venv
source .venv/bin/activate            # run this in every new terminal window
pip install --upgrade pip
pip install -e ".[dev]"
cpiv list-causes                     # prints the causes
pytest                               # all tests should pass (about 1-2 minutes)
cpiv demo --out runs/demo            # synthetic end-to-end demo
```

## 4. Publish on GitHub
1. On github.com create a new **empty** repository named `CPIV` (no README, no licence).
2. In Terminal:
```bash
git init && git add . && git commit -m "Initial CPIV release"
git branch -M main
git remote add origin https://github.com/benisobahasa/CPIV.git
git push -u origin main
```
GitHub asks for authentication: use `brew install gh && gh auth login`, or an SSH key, or
a personal access token as the password.

# Real-corpus experiment

## A. Download clean speech (VCTK)
CSTR VCTK Corpus 0.92 (CC BY 4.0, about 11 GB zip):
https://datashare.ed.ac.uk/handle/10283/3443 (DOI 10.7488/ds/2645)
```bash
mkdir -p ~/data/VCTK && cd ~/data/VCTK
# download the main zip from the page above (browser), then:
unzip ~/Downloads/VCTK-Corpus-0.92.zip -d ~/data/VCTK
ls ~/data/VCTK           # note the folder with one sub-folder per speaker, e.g. p225, p226...
```
Open the README in the zip to confirm the folder names. Speaker folders contain `*_mic1.flac`
and `*_mic2.flac` (two microphones); CPIV only needs one.

## B. Build the benchmark (start small)
```bash
cd ~/Projects/CPIV && source .venv/bin/activate
cpiv build-benchmark --sources ~/data/VCTK/<folder_with_speakers> \
    --pattern "*_mic1.flac" --max-files 100 --out runs/vctk/data
```
Each speaker folder is one group, so train and test speakers never overlap. 100 files
produce a few thousand samples; expect tens of minutes on a laptop. Scale to 300+ files later.
For LibriSpeech (layout `speaker/chapter/utt.flac`) add `--group-depth 1`.

## C. Train and read the report
```bash
cpiv benchmark --data runs/vctk/data --out runs/vctk/paired --mode paired
cpiv benchmark --data runs/vctk/data --out runs/vctk/blind  --mode blind
open runs/vctk/paired/report.md
```
Check: F1 per protocol, which causes are confused, novelty AUROC, faithfulness gaps.

## D. Repeat for statistics (needed before any claim)
```bash
cpiv repeat --sources ~/data/VCTK/<folder_with_speakers> --pattern "*_mic1.flac" \
    --max-files 100 --out runs/vctk/repeat --seeds 0 1 2 3 4 --modes paired blind --backends rf logreg
```

## E. Test on real-world audio (not simulated)
Datasets with real channels (check each licence and README first):

| Dataset | Why | Link |
|---|---|---|
| Device Recorded VCTK (DR-VCTK) | Clean VCTK utterances re-recorded on consumer devices in offices: real reverberation, noise and device response, with clean references | https://datashare.ed.ac.uk/handle/10283/3038 |
| ASVspoof 2017 v2 | Real replay recordings (many rooms and devices), derived from RedDots | https://datashare.ed.ac.uk/handle/10283/3055 |
| ASVspoof 2019 (LA and PA) | Voice-conversion/TTS attacks (LA, 7 GB) and simulated replay (PA, 16 GB), derived from VCTK | https://datashare.ed.ac.uk/handle/10283/3336 |
| ReMASC | Real replay in everyday environments (paper: https://arxiv.org/abs/1904.03365; get the data via the link in the paper) | see paper |

Use with CPIV today:
```bash
# real clean vs. real re-recorded pairs (matched by file name)
python examples/real_pairs_check.py runs/vctk/paired/model.joblib CLEAN_DIR RECORDED_DIR runs/dr_vctk.csv
```
Then check whether `replay`, `reverberation` and `additive_noise` receive high probability and
whether `novelty` stays low. If the model fails on real recordings, that is a result:
report it and improve the simulations. ASVspoof files carry their own genuine/spoof labels,
not CPIV cause chains, so use them for qualitative checks and, with your own adapter, as
external test sets.

For real voice conversion, wrap an open-source system with `ExternalVoiceConversion`
(`examples/plug_in_voice_conversion.py`) and rebuild the benchmark.
