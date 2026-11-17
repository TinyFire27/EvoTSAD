# Commands

Run these examples from the experiment workspace. The commands below use the `exp-01.json` example in [Experiment spec](experiment-spec.md). For output layout and directory handoffs, see [Files and artifacts](artifacts.md).

## Discover components

```bash
libraryts ls algorithms --task anomaly_detection
libraryts ls algorithms --task anomaly_detection --name KMeansAD
libraryts ls preprocessors --task anomaly_detection
libraryts ls preprocessors --task anomaly_detection --name StandardScaler
```

Both subcommands require `--task`. Without `--name`, the result is a JSON list of class names and descriptions. With `--name`, use the exact short class name shown in the list. The detail object includes `class`, `task`, `description`, and `params`. A value of `"required"` marks a parameter you must supply. A `python_repr` object describes a Python default that can be left to the constructor.

Detector details also include `core_params`, the parameter names to consider first when tuning, and `recommended_preprocessors`, suggested pipeline steps in execution order. Review the suggested steps and add the applicable ones to `estimator` in the spec.

## Run a complete experiment

```bash
libraryts run --spec exp-01.json --output-dir outputs/exp-01
```

Use `run` when you have separate training and labeled evaluation CSVs and want all three stages. It returns the metrics and paths to the fitted model, scores, and evaluation.

## Run individual stages

```bash
libraryts train --spec exp-01.json --output-dir outputs/train-01
libraryts predict --spec exp-01.json --model-dir outputs/train-01 --output-dir outputs/predict-01
libraryts evaluate --spec exp-01.json --prediction-dir outputs/predict-01 --output-dir outputs/evaluate-01
```

- **`train`** fits and saves the configured detector or pipeline. Use it on its own when the deliverable is a trained model.
- **`predict`** loads the fitted model or pipeline from `--model-dir` and writes one anomaly score per input row. The scoring CSV can be unlabeled.
- **`evaluate`** loads `scores.csv` from `--prediction-dir` and labels from the evaluation CSV identified by `data.test`. It computes the metrics in the spec without rerunning the model.

Each example uses a fresh output directory. `--spec` is required for all four commands above. Use `libraryts COMMAND --help` to check a command's options.

## Interpret failures

| Exit code | stdout | stderr |
|---|---|---|
| `0` | One result JSON object or array | May contain training progress or other logs |
| `1` | JSON with an `error` message | May contain logs and, with `--debug`, the traceback |
| `2` | Usually empty | Command usage error, such as a missing option |

Capture stdout and stderr separately. Put the root `--debug` option before the subcommand when you need a traceback, for example `libraryts --debug run --spec exp-01.json --output-dir outputs/exp-01-debug`.
