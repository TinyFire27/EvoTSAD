# Experiment Protocol

## Initialize the Task

Confirm the user's objective, data paths, and maximum number of trials. Then call `task_init` once to initialize a new task and create its `.evotsad/` directory and task files. The tool automatically trains five baseline detectors on `data.train` using their default parameters and reports their metrics on `data.eval`. Use these metrics as reference for subsequent trials. The full trial budget remains available after this step. Call `task_init` again only to start a new task or intentionally reset the current one, since it clears the existing task directory.

### Data Requirements

The tools accept time-series data in CSV format only.

- `data.train` is required. Use it to train detectors and analyze the series.
- `data.val` is an optional validation segment for models that support validation or early stopping during training. If the user has not provided one and you determine that it is needed, you may split off the final portion of the training series for validation.
- `data.eval` is the evaluation segment for comparing candidate pipelines across trials. It must be a different CSV file from `data.train` and must contain a `label` column.

## Profile the Time Series

After `task_init` succeeds, use `bash` to profile `data.train` before submitting the first trial. The goal is to identify properties of the series that can guide your choice of preprocessing, detector, or hyperparameters. Choose analyses relevant to the series and the task objective. Use `write_file` to save your findings to `.evotsad/profile.json` in the following format. Replace the field descriptions with actual values and include at least one observation.

```json
{
  "length": "Integer. Total number of rows in data.train.",
  "missing_ratio": "Number between 0 and 1. Fraction of rows where value is missing.",
  "observations": [
    {
      "finding": "A property or pattern you observed in the series.",
      "evidence": "A measurement or specific feature of the data that supports the finding.",
      "implication": "What the finding suggests about preprocessing, detector choice, or hyperparameters, and why."
    }
  ]
}
```

During the experiment, you can revisit the series and use `edit_file` to update `profile.json` as new evidence emerges.

## Run and Refine Candidate Pipelines

1. **Propose the first pipeline.** Use the observations in `.evotsad/profile.json` and the baseline scores from `task_init` to choose preprocessing steps, a detector, and its parameters. In `reason`, explain what you want this trial to test.
2. **Run a trial.** Pass the candidate as the `pipeline` argument to `run_trial`. The tool trains and evaluates the complete pipeline. A short `reason` is recommended to explain why the candidate is worth trying or what you want to test in this trial.
3. **Choose the next trial.** Review the current trial's metrics and the best trial so far. Consider the candidates you have already evaluated. You can tune a promising detector's core hyperparameters, revise the preprocessing steps, or try another detector family. Base each change on an observed characteristic of the series or a result from a previous trial.
4. **Handle failures.** If a tool returns a correctable error, use the error message to revise the candidate and resubmit it under the same task. Only successful evaluations count against the trial budget.

## Workspace and Experiment Artifacts

The experiment tools automatically manage task artifacts, including those from each trial, under task directory `.evotsad/` in the current workspace. You create and maintain the series profile at `.evotsad/profile.json`.

```text
.evotsad/
├── task.json                    # Configuration shared by all trials
├── baselines.json               # Default-parameter baseline scores and any errors, written by task_init
├── profile.json                 # Training-series profile created and updated by the agent
├── history.jsonl                # Rationale, metrics, and artifact paths for each successful trial
└── experiments/                 # Per-trial directories
    └── exp-001/                 # Automatically assigned, unique trial ID
        ├── spec.json            # Configuration used to run this trial
        ├── model.zip            # Trained detector or pipeline
        ├── scores.csv           # Pointwise anomaly scores on data.eval
        ├── metrics.json         # Evaluation metrics for this trial
        └── manifest.json        # Completion time and file names relative to this directory
```

The returned `experiment_dir` is the trial directory itself. Use `get_history()` to compare earlier candidates, `get_experiment(experiment_id)` to revisit one, and `get_best_experiment()` to retrieve the current leader with its full spec.

## Stop and Deliver

Stop when you have used the trial budget, reached the target agreed with the user, or received a request to stop. Select the trial with the best primary metric among all evaluated candidates. Report its ID and score from `best_experiment`, link its configuration at `best_experiment.spec_path` and its trained model at `<best_experiment.experiment_dir>/model.zip`, and briefly explain your selection.
