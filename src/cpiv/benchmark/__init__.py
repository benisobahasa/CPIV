"""CPIV benchmark: generation, protocols, metrics, runner."""

from .generate import (
    build_dataset,
    collect_sources,
    load_dataset,
    select_mode,
    split_indices,
    synthetic_sources,
)
from .metrics import attribution_metrics, open_set_metrics
from .repeat import repeat_benchmark, repeat_markdown
from .run import evaluate_model, save_report, to_markdown, train_and_evaluate
from .score import score_predictions

__all__ = ["build_dataset", "collect_sources", "synthetic_sources", "load_dataset", "select_mode",
           "split_indices", "attribution_metrics", "open_set_metrics", "train_and_evaluate",
           "to_markdown", "save_report", "evaluate_model", "repeat_benchmark", "repeat_markdown",
           "score_predictions"]
