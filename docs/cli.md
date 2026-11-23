# CLI reference

| Command | Purpose |
|---|---|
| `cpiv list-causes` | Show known causes and unknown manipulations |
| `cpiv perturb --input a.wav --output b.wav --cause replay --cause additive_noise --severity 0.6` | Apply interventions to one file |
| `cpiv build-benchmark --sources DIR --out OUT` (or `--synthetic`) | Generate audio, `manifest.jsonl`, `features.npz` |
| `cpiv benchmark --data OUT/data --out OUT/run --mode paired\|blind --backend rf\|hgb\|logreg` | Train and evaluate all protocols; writes `report.md`, `results.json`, `model.joblib` |
| `cpiv repeat --sources DIR --out OUT --seeds 0 1 2 3 4 --modes paired blind --backends rf logreg` | Repeated runs with mean, std, 95% CI |
| `cpiv evaluate-external --model model.joblib --data OTHER_CORPUS/data` | Cross-corpus evaluation |
| `cpiv score --data OUT/data --predictions preds.csv` | Score any third-party system |
| `cpiv attribute --model model.joblib --test t.wav --reference r.wav [--json]` | Explain a recording pair |
| `cpiv demo` | End-to-end demo on synthetic voices |

Predictions CSV for `cpiv score`: header `id,<each cause name>[,unknown_score]`, one row per
manifest id, probabilities in [0, 1].
