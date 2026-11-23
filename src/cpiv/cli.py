"""Command line interface: ``cpiv --help``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .config import CAUSES, UNKNOWN_CAUSES


def _build(args) -> dict:
    from .benchmark import build_dataset, collect_sources, synthetic_sources

    if args.sources:
        items = collect_sources(args.sources, max_files=args.max_files, seed=args.seed,
                                pattern=args.pattern, group_depth=args.group_depth)
        if not items:
            sys.exit(f"no audio files found in {args.sources}")
    else:
        items = synthetic_sources(args.speakers, args.utts, seed=args.seed)
    return build_dataset(items, out_dir=args.out, paired=not args.blind, seed=args.seed,
                         per_cause=args.per_cause, n_jobs=args.jobs)


def cmd_list(_args) -> None:
    print("Known causes (trainable):")
    for c in CAUSES:
        print(f"  {c}")
    print("Unknown manipulations (open-set test only):")
    for c in UNKNOWN_CAUSES:
        print(f"  {c}")


def cmd_perturb(args) -> None:
    from .audio import load_audio, save_audio
    from .interventions import apply_chain

    y = load_audio(args.input)
    rng = np.random.default_rng(args.seed)
    out, log = apply_chain(y, 16000, args.cause, rng, args.severity)
    save_audio(args.output, out)
    print(json.dumps(log, indent=2, default=float))


def cmd_build(args) -> None:
    data = _build(args)
    print(f"[cpiv] wrote {len(data['meta'])} samples to {args.out}")


def cmd_benchmark(args) -> None:
    from .benchmark import load_dataset, save_report, to_markdown, train_and_evaluate

    data = load_dataset(args.data)
    res = train_and_evaluate(data, mode=args.mode, backend=args.backend, seed=args.seed,
                             n_estimators=args.trees, model_path=Path(args.out) / "model.joblib")
    save_report(res, args.out)
    print(to_markdown(res))


def cmd_evaluate_external(args) -> None:
    """Cross-corpus protocol: evaluate a trained model on a dataset built from another corpus."""
    from .attribution import CausalAttributor
    from .benchmark import evaluate_model, load_dataset

    res = evaluate_model(CausalAttributor.load(args.model), load_dataset(args.data))
    print(json.dumps(res, indent=2, default=float))


def cmd_score(args) -> None:
    """Score a third-party system from a predictions CSV (see docs/benchmark_protocol.md)."""
    from .benchmark import score_predictions

    print(json.dumps(score_predictions(args.data, args.predictions, args.threshold), indent=2, default=float))


def cmd_repeat(args) -> None:
    from .benchmark import collect_sources, repeat_benchmark, repeat_markdown, synthetic_sources

    items = (collect_sources(args.sources, max_files=args.max_files, pattern=args.pattern,
                             group_depth=args.group_depth) if args.sources
             else synthetic_sources(args.speakers, args.utts, seed=0))
    rep = repeat_benchmark(items, args.seeds, modes=args.modes, backends=args.backends,
                           n_estimators=args.trees, per_cause=args.per_cause)
    md = repeat_markdown(rep)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "repeat_report.md").write_text(md)
    (out / "repeat_results.json").write_text(json.dumps(rep, indent=2, default=float))
    print(md)


def cmd_attribute(args) -> None:
    from .api import attribute_pair

    res = attribute_pair(args.model, args.test, args.reference, top_k=args.top_k)
    print(json.dumps(res.to_dict(), indent=2) if args.json else res.text())


def cmd_demo(args) -> None:
    """Self-contained demo on synthetic voices: build -> train -> evaluate -> explain."""
    from .api import attribute_pair
    from .audio import save_audio, synth_voice
    from .benchmark import (
        build_dataset,
        save_report,
        synthetic_sources,
        to_markdown,
        train_and_evaluate,
    )
    from .interventions import apply_chain

    out = Path(args.out)
    items = synthetic_sources(args.speakers, 2, seed=args.seed, duration=2.5)
    data = build_dataset(items, out_dir=out / "data", seed=args.seed, per_cause=1, n_jobs=args.jobs)
    res = train_and_evaluate(data, n_estimators=150, model_path=out / "model.joblib", seed=args.seed)
    save_report(res, out)
    print(to_markdown(res))

    ref = synth_voice(seed=12345, duration=2.5)
    rng = np.random.default_rng(7)
    test, _ = apply_chain(ref, 16000, ["replay", "additive_noise"], rng, severity=0.5)
    save_audio(out / "demo_ref.wav", ref)
    save_audio(out / "demo_test.wav", test)
    print("\n=== Explanation for a replay + noise example ===")
    print(attribute_pair(res["_model"], test, ref).text())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cpiv", description="Causal Perturbation & Intervention for Voice")
    p.add_argument("--version", action="version", version=f"cpiv {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list-causes", help="show known causes and unknown manipulations")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("perturb", help="apply controlled interventions to one audio file")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--cause", action="append", required=True, choices=CAUSES + UNKNOWN_CAUSES)
    s.add_argument("--severity", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_perturb)

    s = sub.add_parser("build-benchmark", help="generate benchmark audio + features")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--sources", help="folder of clean audio (one sub-folder per speaker)")
    g.add_argument("--synthetic", action="store_true", help="use built-in synthetic voices")
    s.add_argument("--out", required=True)
    s.add_argument("--speakers", type=int, default=12)
    s.add_argument("--utts", type=int, default=3)
    s.add_argument("--max-files", type=int, default=None)
    s.add_argument("--pattern", default=None, help="file-name glob, e.g. '*_mic1.flac'")
    s.add_argument("--group-depth", type=int, default=0,
                   help="0: speaker = parent folder; N: speaker = first N path parts under --sources")
    s.add_argument("--per-cause", type=int, default=2)
    s.add_argument("--blind", action="store_true", help="no reference recordings (blind mode)")
    s.add_argument("--jobs", type=int, default=-1)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("benchmark", help="train on a built benchmark and report all protocols")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=["paired", "blind"], default="paired")
    s.add_argument("--backend", choices=["rf", "hgb", "logreg"], default="rf")
    s.add_argument("--trees", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("attribute", help="explain what changed between two recordings")
    s.add_argument("--model", required=True)
    s.add_argument("--test", required=True)
    s.add_argument("--reference", default=None)
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_attribute)

    s = sub.add_parser("evaluate-external", help="evaluate a trained model on another corpus' dataset")
    s.add_argument("--model", required=True)
    s.add_argument("--data", required=True)
    s.set_defaults(fn=cmd_evaluate_external)

    s = sub.add_parser("score", help="score a third-party system from a predictions CSV")
    s.add_argument("--data", required=True, help="built benchmark folder with manifest.jsonl")
    s.add_argument("--predictions", required=True)
    s.add_argument("--threshold", type=float, default=0.5)
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("repeat", help="repeat the benchmark over seeds with confidence intervals")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--sources")
    g.add_argument("--synthetic", action="store_true")
    s.add_argument("--out", required=True)
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--modes", nargs="+", choices=["paired", "blind"], default=["paired", "blind"])
    s.add_argument("--backends", nargs="+", choices=["rf", "hgb", "logreg"], default=["rf"])
    s.add_argument("--speakers", type=int, default=12)
    s.add_argument("--utts", type=int, default=2)
    s.add_argument("--max-files", type=int, default=None)
    s.add_argument("--pattern", default=None, help="file-name glob, e.g. '*_mic1.flac'")
    s.add_argument("--group-depth", type=int, default=0,
                   help="0: speaker = parent folder; N: speaker = first N path parts under --sources")
    s.add_argument("--per-cause", type=int, default=2)
    s.add_argument("--trees", type=int, default=200)
    s.set_defaults(fn=cmd_repeat)

    s = sub.add_parser("demo", help="end-to-end demo on synthetic voices")
    s.add_argument("--out", default="runs/demo")
    s.add_argument("--speakers", type=int, default=10)
    s.add_argument("--jobs", type=int, default=-1)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
