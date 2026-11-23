#!/usr/bin/env bash
# Example: build and evaluate the benchmark on a local copy of a real corpus.
# Corpus layout: <root>/<speaker_id>/<utterance>.wav (one folder per speaker).
# Check the corpus licence before redistributing any generated audio.
set -euo pipefail
ROOT="${1:?usage: $0 /path/to/corpus [out_dir]}"
OUT="${2:-runs/real}"
cpiv build-benchmark --sources "$ROOT" --max-files 600 --out "$OUT/data" --per-cause 2
cpiv benchmark --data "$OUT/data" --out "$OUT/paired" --mode paired
cpiv benchmark --data "$OUT/data" --out "$OUT/blind" --mode blind
