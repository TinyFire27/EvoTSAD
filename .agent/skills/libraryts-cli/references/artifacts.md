# Files and Artifacts

## Choose an output directory

Pass a fresh directory to `--output-dir` for each `train`, `predict`, `evaluate`, or `run`. LibraryTS creates the directory and any missing parents. Relative output paths resolve from the command's working directory. Use a distinct directory for each experiment and retain the input spec so the experiment can be rerun.

## Files produced by each stage

| Command | Files under `--output-dir` | Success JSON fields |
|---|---|---|
| `train` | `model.zip`, `manifest.json` | `model_dir`, `model_path`, `manifest_path` |
| `predict` | `scores.csv`, `manifest.json` | `prediction_dir`, `scores_path`, `n_scores`, `manifest_path` |
| `evaluate` | `metrics.json`, `manifest.json` | `evaluation_dir`, `metrics_path`, `metrics`, `manifest_path` |
| `run` | The three stage directories and a top-level `manifest.json` | `run_dir`, `model_dir`, `prediction_dir`, `evaluation_dir`, `metrics`, `manifest_path` |

A complete run at `outputs/exp-01` has this layout:

```text
outputs/exp-01/
├── manifest.json
├── model/
│   ├── model.zip
│   └── manifest.json
├── prediction/
│   ├── scores.csv
│   └── manifest.json
└── evaluation/
    ├── metrics.json
    └── manifest.json
```

The top-level run manifest records the experiment spec, the stage directories, and the metric values. `scores.csv` has `timestamp` and `score` columns in input order. A higher score indicates a more anomalous point. `metrics.json` contains the requested metric values.

## Reuse completed stages

- Pass the returned `model_dir` to `predict --model-dir`. This is the directory containing `model.zip`, which includes the fitted preprocessing steps. Load models from trusted sources.
- Pass the returned `prediction_dir` to `evaluate --prediction-dir`. This is the directory containing `scores.csv`. Use the matching evaluation CSV in the spec.
- Read metric values from the success JSON or `metrics.json`. Keep the returned `manifest_path` to identify the experiment and its configuration. Returned paths are absolute.

If `run` fails partway through, completed stage files may still be present. Check the error and the available artifacts. Retry the whole run in a fresh directory, or reuse a successfully completed stage with a standalone command and a fresh output directory.
