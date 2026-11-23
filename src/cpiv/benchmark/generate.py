"""Benchmark data generation: clean sources -> controlled interventions -> features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from ..audio import load_audio, peak_normalize, save_audio, synth_voice
from ..config import CAUSES, SAMPLE_RATE
from ..features import feature_vector
from ..interventions import apply_chain
from .spec import Condition, conditions_for_source

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


# ----------------------------------------------------------------------------- sources
def collect_sources(root: str | Path, max_files: int | None = None, seed: int = 0,
                    pattern: str | None = None, group_depth: int = 0) -> list[dict]:
    """Scan a folder of clean recordings.

    * ``pattern``: optional glob on the file name, e.g. ``"*_mic1.flac"`` (VCTK).
    * ``group_depth``: 0 = speaker is the parent folder (VCTK layout). N >= 1 = speaker is the
      first N path components under ``root`` (LibriSpeech: ``group_depth=1`` so that the speaker
      folder, not the chapter folder, is the group). Splits are speaker-disjoint on this group.
    """
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT
                   and (pattern is None or p.match(pattern)))
    if max_files and len(files) > max_files:
        rng = np.random.default_rng(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), max_files, replace=False))]

    def group_of_file(p: Path) -> str:
        if group_depth <= 0:
            return p.parent.name
        return "/".join(p.relative_to(root).parts[:group_depth])

    return [{"id": p.stem, "group": group_of_file(p), "path": str(p)} for p in files]


def synthetic_sources(n_speakers: int = 12, utts_per_speaker: int = 3, seed: int = 0,
                      duration: float = 3.0) -> list[dict]:
    rng = np.random.default_rng(seed)
    items = []
    for s in range(n_speakers):
        f0, fs = float(rng.uniform(90, 240)), float(rng.uniform(0.85, 1.2))
        for u in range(utts_per_speaker):
            items.append({"id": f"syn_s{s:02d}_u{u}", "group": f"syn_s{s:02d}",
                          "synth": {"seed": int(rng.integers(0, 2**31 - 1)), "f0": f0, "fs": fs,
                                    "duration": duration}})
    return items


def assign_source_splits(items: list[dict], frac=(0.6, 0.1, 0.3), seed: int = 0) -> dict[str, str]:
    """Group-disjoint train/val/test assignment (deterministic)."""
    groups = sorted({it["group"] for it in items})
    order = sorted(groups, key=lambda g: hashlib.md5(f"{seed}:{g}".encode()).hexdigest())
    n = len(order)
    n_test = max(1, round(frac[2] * n)) if n >= 3 else 1
    n_val = max(1, round(frac[1] * n)) if n >= 5 else 0
    mapping = {}
    for i, g in enumerate(order):
        mapping[g] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    if "train" not in mapping.values():
        mapping[order[-1]] = "train"
    return mapping


def _load_source(it: dict, max_seconds: float, sr: int) -> np.ndarray:
    if "synth" in it:
        s = it["synth"]
        y = synth_voice(s["seed"], s["duration"], sr, s["f0"], s["fs"])
    else:
        y = load_audio(it["path"], sr)
    y = y[: int(max_seconds * sr)]
    return peak_normalize(y, 0.8)


# --------------------------------------------------------------------------- generation
def _process(it: dict, ci: int, cond: Condition, seed: int, paired: bool, sr: int,
             max_seconds: float, out_dir: Path | None, kwargs: dict) -> dict | None:
    try:
        ref = _load_source(it, max_seconds, sr)
        h = int(hashlib.md5(f"{seed}:{it['id']}:{ci}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        if cond.causes:
            test, log = apply_chain(ref, sr, cond.causes, rng, cond.severity, **kwargs)
        else:
            test, log = ref * 1.0, []
        uid = f"{it['id']}__c{ci:03d}"
        if out_dir is not None:
            save_audio(out_dir / "audio" / f"{uid}.wav", test, sr)
            if paired:
                ref_path = out_dir / "audio" / f"{it['id']}__ref.wav"
                if not ref_path.exists():
                    save_audio(ref_path, ref, sr)
        x, names = feature_vector(test, ref if paired else None, sr)
        y = np.zeros(len(CAUSES), dtype=int)
        for c in cond.causes:
            if c in CAUSES:
                y[CAUSES.index(c)] = 1
        return {"x": x, "names": names, "y": y,
                "meta": {"id": uid, "source": it["id"], "group": it["group"], "split": cond.split,
                         "causes": cond.causes, "severity": cond.severity, "unknown": cond.unknown,
                         "params": log}}
    except Exception as e:  # keep the benchmark robust to single bad files
        return {"error": f"{it.get('id')}#{ci}: {e}"}


def build_dataset(items: list[dict], out_dir: str | Path | None = None, paired: bool = True,
                  seed: int = 0, per_cause: int = 2, n_pairs: int = 2, n_unknown: int = 1,
                  max_seconds: float = 6.0, n_jobs: int = -1, sr: int = SAMPLE_RATE,
                  intervention_kwargs: dict | None = None, verbose: bool = True) -> dict:
    """Generate the labelled benchmark and its feature matrix.

    If ``out_dir`` is given, audio files, a reference copy and ``manifest.jsonl`` are written
    (so other researchers can evaluate their own systems on the exact same audio).
    """
    out = Path(out_dir) if out_dir else None
    split_map = assign_source_splits(items, seed=seed)
    rng = np.random.default_rng(seed)
    jobs = []
    for it in items:
        conds = conditions_for_source(split_map[it["group"]], rng, per_cause=per_cause,
                                      n_pairs=n_pairs, n_unknown=n_unknown)
        for ci, c in enumerate(conds):
            jobs.append((it, ci, c))
    if verbose:
        print(f"[cpiv] generating {len(jobs)} samples from {len(items)} sources "
              f"({len(set(split_map.values()))} source partitions)")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process)(it, ci, c, seed, paired, sr, max_seconds, out, intervention_kwargs or {})
        for it, ci, c in jobs)
    errors = [r["error"] for r in results if r and "error" in r]
    good = [r for r in results if r and "error" not in r]
    if verbose and errors:
        print(f"[cpiv] {len(errors)} samples skipped, e.g. {errors[0]}")
    if not good:
        raise RuntimeError("no samples generated")
    data = {"X": np.stack([r["x"] for r in good]), "Y": np.stack([r["y"] for r in good]),
            "feature_names": good[0]["names"], "meta": [r["meta"] for r in good],
            "paired": paired, "seed": seed}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "manifest.jsonl", "w") as f:
            for m in data["meta"]:
                f.write(json.dumps(m, default=float) + "\n")
        np.savez_compressed(out / "features.npz", X=data["X"], Y=data["Y"],
                            names=np.array(data["feature_names"]), paired=paired)
        with open(out / "causes.json", "w") as f:
            json.dump({"causes": CAUSES}, f, indent=2)
    return data


def load_dataset(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    z = np.load(out / "features.npz", allow_pickle=False)
    meta = [json.loads(line) for line in open(out / "manifest.jsonl")]
    return {"X": z["X"], "Y": z["Y"], "feature_names": list(z["names"]), "meta": meta,
            "paired": bool(z["paired"])}


def split_indices(data: dict) -> dict[str, np.ndarray]:
    sp = np.array([m["split"] for m in data["meta"]])
    return {s: np.flatnonzero(sp == s) for s in np.unique(sp)}


def select_mode(data: dict, mode: str) -> tuple[np.ndarray, list[str]]:
    """Choose feature columns: 'paired' (delta + test) or 'blind' (test only)."""
    names = data["feature_names"]
    if mode == "paired":
        if not data["paired"]:
            raise ValueError("dataset was built without references")
        return data["X"], names
    cols = [i for i, n in enumerate(names) if n.startswith("test::")]
    return data["X"][:, cols], [names[i] for i in cols]
