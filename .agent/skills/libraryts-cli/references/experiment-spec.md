# Experiment Spec

Pass the path to a UTF-8 JSON file with `--spec`. Use the working directory of the CLI invocation as the base for relative paths, including paths inside the JSON. Keep the same working directory across stages when reusing a spec with relative paths.

## Fields by command

| Command | Required spec fields | Optional spec fields |
|---|---|---|
| `train` | `task`, `data.train`, `estimator` | `data.val`, `seed`, `fit_params` |
| `predict` | `task`, `data.test` | |
| `evaluate` | `task`, `data.test`, `metrics` | |
| `run` | `task`, `data.train`, `data.test`, `estimator`, `metrics` | `data.val`, `seed`, `fit_params` |

- Set `task` to `"anomaly_detection"`.
- Use `data.train` for fitting, `data.val` for optional validation during fitting, and `data.test` for the series to score or evaluate in this invocation. For `run`, provide different files for `data.train` and `data.test`, and use a labeled `data.test`. The task protocol determines which data and labels the Agent can use.
- Write `estimator` as an ordered array, even for a single detector. Each step needs a `class` with the exact short class name. Put any preprocessors before the detector. Use the optional `params` object for constructor parameters that the component requires or that you want to override. Query current defaults with `libraryts ls ... --name CLASS`. Include the applicable preprocessing steps explicitly.
- Use the optional top-level `seed` to override `seed` or `random_state` on components that accept either. Use `fit_params.show_progress` for progress output on supported detectors. Early stopping settings belong in the detector step's `params`.
- Set `metrics` to an object mapping metric names to parameter objects. An empty parameter object uses the metric defaults. For example, `{"VUS-PR": {"sliding_window": 100}, "Point-F1": {}}` requests two metrics. `run` needs at least one metric.

## Local CSV format

- Include a header and one numeric value column for a univariate series, such as `value`. An optional `timestamp` column supplies the index. Without it, row numbers are used. Prepare the rows in time order.
- For evaluation, put the binary ground-truth `label` column in the same `data.test` CSV as the values. Use `0` for normal and `1` for anomalous. VUS-PR requires both classes. Training and scoring use the value columns and exclude `label` from the model inputs.
- Evaluate scores against the same CSV used for prediction, with matching row order and index. A registered dataset ID can refer to an unsplit CSV in standalone commands. Use explicit local split paths for `run`.

## Minimal complete example

Save this as `exp-01.json` in the working directory. The example uses `StandardScaler` and `KMeansAD`. Use a training CSV with at least 16 rows and a scoring CSV with at least 8 rows and both label classes. Adjust paths, window length, and metric parameters to match the actual task.

```json
{
  "task": "anomaly_detection",
  "data": {
    "train": "data/train.csv",
    "test": "data/test.csv"
  },
  "estimator": [
    {"class": "StandardScaler"},
    {"class": "KMeansAD", "params": {"win_len": 8, "k": 2}}
  ],
  "seed": 2026,
  "metrics": {"VUS-PR": {"sliding_window": 8}}
}
```
