# Tool Use

## General Workspace Tools

- `bash(command)` runs a command in the workspace. Use it to inspect data or to write and run analysis scripts.
- `read_file(path)` reads a UTF-8 text file in the workspace. Check `truncated` to see whether the tool returned only part of the file.
- `read_image(path)` displays a PNG, JPEG, or WebP image from the workspace.
- `write_file(path, content)` creates a UTF-8 text file in the workspace.
- `edit_file(path, old_text, new_text)` replaces one exact match in an existing UTF-8 workspace file; use it to revise `.evotsad/profile.json`.

## Time-Series Anomaly Detection Toolkit

- `search_memory(query?, limit?)` searches experiences for the selected memory mode in this workspace. An empty query ranks by the current training profile's 17 fields, or recency when no profile is available. The selected mode determines what is saved; in `none` mode nothing is saved or retrieved. Inspect relevant past pipelines when planning a task, but validate them on the current dataset; scores from different tasks are not directly comparable.

The built-in toolkit provides detectors and preprocessors and handles pipeline training, scoring, and evaluation. Run experiments through the interfaces below. Follow their argument specifications. You can use the tools without knowing how they are implemented.

- `task_init(objective, task_description, data, max_trials, primary_metric?, seed?)` starts a new task from the user's requirements. It initializes `.evotsad/`, writes the task configuration and training profile to `.evotsad/profile.json`, and evaluates five baseline detectors on `data.eval` without using the trial budget. Windowed detectors receive a data-length-aware window; the results include their actual parameters, metrics, or errors and are stored under `baselines` in `profile.json`. Provide the complete original request and amendments in task_description. Initial memories are also returned. Finish the current task before starting a new one.
- `list_components(kind, name?)` looks up components. Set `kind` to `detectors` or `preprocessors` and omit `name` to list the available components of that type. To configure a pipeline step, pass its class name as `name` to look up its parameters and their defaults. Detector details also list the main hyperparameters to tune and the preprocessors recommended for use before the detector.
- `run_trial(pipeline, reason?)` is the entry point for each trial. The tool trains the pipeline using the task settings, then scores and evaluates it on `data.eval`. A successful evaluation counts as one trial. The result includes metrics, budget progress, and the best trial so far. `spec_path` points to the configuration file; `experiment_dir` contains the trained model, scores, metrics, and manifest. Reaching the budget automatically ends the whole task and writes its memory.
- `get_history()` returns all successful trial records in `trial_history`, in recorded order; it returns `[]` before the first successful trial.
- `get_experiment(experiment_id)` returns the full history entry for an exact ID such as `exp-002` in `experiment`.
- `get_best_experiment()` returns the full record with the highest primary metric in `best_experiment`, or `null` before the first successful trial. Ties select the earliest trial. Baselines are not included.

- `finish_task(reason?)` ends the whole task early and publishes memory when updates are enabled. With updates disabled, it ends the task without writing memory while retrieval remains available. Call before delivering final results if the budget has not already ended the task. It does not run any new training.

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
