"""Run a trained CPIV model on real (reference, re-recorded) file pairs and save a CSV.

Typical use: clean recordings vs. the same utterances replayed and re-recorded on a device
(e.g. Device Recorded VCTK). Files are matched by file-name stem; edit `stem_of` if the two
folders name their files differently.

    python examples/real_pairs_check.py runs/vctk/paired/model.joblib CLEAN_DIR RECORDED_DIR out.csv
"""

import csv
import sys
from pathlib import Path

from cpiv.api import attribute_pair
from cpiv.attribution import CausalAttributor
from cpiv.benchmark.generate import AUDIO_EXT


def stem_of(path: Path) -> str:
    return path.stem


def main(model_path: str, clean_dir: str, rec_dir: str, out_csv: str, limit: int = 200) -> None:
    model = CausalAttributor.load(model_path)
    clean = {stem_of(p): p for p in Path(clean_dir).rglob("*") if p.suffix.lower() in AUDIO_EXT}
    rec = {stem_of(p): p for p in Path(rec_dir).rglob("*") if p.suffix.lower() in AUDIO_EXT}
    stems = sorted(set(clean) & set(rec))[:limit]
    if not stems:
        sys.exit("no matching file names; adjust stem_of()")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", *model.causes, "novelty"])
        for s in stems:
            r = attribute_pair(model, str(rec[s]), str(clean[s]))
            w.writerow([s, *[f"{c.probability:.3f}" for c in r.causes], f"{r.novelty:.2f}"])
    print(f"wrote {len(stems)} rows to {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:5])
