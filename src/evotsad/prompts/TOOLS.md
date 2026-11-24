# Tool Use

## General Workspace Tools

- `bash(command)` runs a command in the workspace. Use it to inspect data or to write and run analysis scripts.
- `read_file(path)` reads a UTF-8 text file in the workspace. Check `truncated` to see whether the tool returned only part of the file.
- `read_image(path)` displays a PNG, JPEG, or WebP image from the workspace.
- `write_file(path, content)` creates a UTF-8 text file in the workspace.

## Time-Series Anomaly Detection Toolkit

The built-in toolkit provides detectors and preprocessors and handles pipeline training, scoring, and evaluation. Run experiments through the interfaces below. Follow their argument specifications. You can use the tools without knowing how they are implemented.

- `task_init(objective, data, max_trials, primary_metric?, seed?)` starts a new task from the user's requirements. It initializes the workspace task directory `.evotsad/` and writes the objective, data paths, trial budget, primary metric, and random seed to `task.json`. All trials in the task share these settings. On success, the tool returns the task configuration and its paths. Calling it again resets the current task and its existing artifacts.
- `list_components(kind, name?)` looks up components. Set `kind` to `detectors` or `preprocessors` and omit `name` to list the available components of that type. To configure a pipeline step, pass its class name as `name` to look up its parameters and their defaults. Detector details also list the main hyperparameters to tune and the preprocessors recommended for use before the detector.
- `run_trial(pipeline, reason?)` is the entry point for each trial. Once you have chosen the components, ordered the preprocessing steps, and set the detector parameters, submit one candidate for training and evaluation through `pipeline`. Use the optional `reason` to state what you want to test. The tool trains the pipeline using the task settings, then scores and evaluates it on `data.eval`. A successful evaluation counts as one trial. The result includes the trial metrics, budget progress, and the best trial so far. `spec_path` points to the configuration file, and `experiment_dir` is the directory containing that file and the trained model, scores, metrics, and manifest.

## How to Specify a `pipeline`

`pipeline` is a nonempty array of steps in execution order. Preprocessing steps are optional. The last step is the detector. Use an array even for a detector alone, for example `[{"class": "Donut"}]`. A recommended preprocessor runs only if you explicitly include it in the pipeline.

Set each step's `class` to the exact component class name. Use `list_components(kind, name)` to look up its parameters and defaults, then put any required parameters and any values you want to change in that step's `params`. Other parameters use their component defaults. You can omit `params` if the defaults cover the entire step. Components that accept a random seed use the task's `seed`.

This pipeline standardizes the series, then applies the `LSTMAD` detector with a window length of 96.

```json
{
  "pipeline": [
    {"class": "StandardScaler"},
    {"class": "LSTMAD", "params": {"win_len": 96}}
  ]
}
```
