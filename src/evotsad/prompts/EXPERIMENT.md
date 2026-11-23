# Experiment Protocol

## Initialize the Task

Confirm the user's objective, the data paths, and the maximum number of trials. Once you have this information, call `task_init` once for the new task to initialize the task directory `.evotsad/` and its files. Calling it again clears and rebuilds the directory. Use it again only to start a new task or reset the current one.

### Data Requirements

The tools accept time-series data in CSV format only.

- `data.train` is required. Use it to train detectors and analyze the series.
- `data.val` is an optional validation segment for models that support validation or early stopping during training. If the user has not provided one and you determine that it is needed, you may split off the final portion of the training series for validation.
- `data.eval` is the evaluation segment for comparing candidate pipelines across trials. It must be a different CSV file from `data.train` and must contain a `label` column.

## Run and Refine Candidate Pipelines

1. **Propose the first pipeline.** Profile the series with the task objective in mind. Use the characteristics you observe to choose preprocessing steps, a detector, and its parameters.
2. **Run a trial.** Pass the candidate as the `pipeline` argument to `run_trial`. The tool trains and evaluates the complete pipeline. A short `reason` is recommended to explain why the candidate is worth trying or what you want to test in this trial.
3. **Choose the next trial.** Review the current trial's metrics and the best trial so far. Consider the candidates you have already evaluated. You can tune a promising detector's core hyperparameters, revise the preprocessing steps, or try another detector family. Base each change on an observed characteristic of the series or a result from a previous trial.
4. **Handle failures.** If a tool returns a correctable error, use the error message to revise the candidate and resubmit it under the same task. Only successful evaluations count against the trial budget.

## Workspace and Experiment Artifacts

The experiment tools automatically manage all task artifacts, including those from each trial, under `.evotsad/` in the current workspace.

```text
.evotsad/
├── task.json                    # Configuration shared by all trials
├── history.jsonl                # Rationale, metrics, and artifact paths for each successful trial
└── experiments/                 # Per-trial directories
    └── exp-001/                 # Automatically assigned, unique trial ID
        ├── spec.json            # Configuration used to run this trial
        └── artifacts/
            ├── model/model.zip          # Trained detector or pipeline
            ├── prediction/scores.csv    # Pointwise anomaly scores on the evaluation segment
            └── evaluation/metrics.json  # Metrics for this trial
```

To inspect a trial later, use the `spec_path` and `output_dir` returned by the tool.

## Stop and Deliver

Stop when you have used the trial budget, reached the target agreed with the user, or received a request to stop. Select the trial with the best primary metric among all evaluated candidates. Use `best_experiment` to report the trial ID, primary metric score, and paths to the experiment configuration and trained model. Briefly explain why you selected that pipeline.
