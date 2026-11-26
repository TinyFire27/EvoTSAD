# Experiment Protocol

## Initialize the Task

Confirm the user's objective, the data paths, and the maximum number of trials. Once you have this information, call `task_init` once for the new task to initialize the task directory `.evotsad/` and its files. Pass task_description containing the full user request, amendments and agreed constraints, not a short summary. Finish the current task before starting another; starting another task replaces its working directory. Keep input CSV contents unchanged throughout a task.

### Data Requirements

The tools accept time-series data in CSV format only.

- `data.train` is required. Use it to train detectors and analyze the series.
- `data.val` is an optional validation segment for models that support validation or early stopping during training. If the user has not provided one and you determine that it is needed, you may split off the final portion of the training series for validation.
- `data.eval` is the evaluation segment for comparing candidate pipelines across trials. It must be a different CSV file from `data.train` and must contain a `label` column.

## Run and Refine Candidate Pipelines

task_init writes `.evotsad/profile.json` with the training-only 17-field profile, `length`, `missing_ratio`, `observations`, and five baseline results. Windowed detectors receive a data-length-aware `win_len`; other parameters use their defaults unless recorded otherwise. It also returns these baselines and initial memories from the selected mode. Baselines are evaluated on `data.eval` outside the trial budget. Before proposing candidates, inspect the profile and memories; use `search_memory` for a more specific query if needed. Prior-task memories are suggestions, not evidence of performance on the current dataset. Use `edit_file` to refine the observations in `profile.json` as new evidence emerges.

1. **Propose the first pipeline.** Use the training profile, baseline metrics, and task objective to choose preprocessing steps, a detector, and its parameters. Explain the trial's purpose in `reason`.
2. **Run a trial.** Pass the candidate as the `pipeline` argument to `run_trial`. The tool trains and evaluates the complete pipeline. A short `reason` is recommended to explain why the candidate is worth trying or what you want to test in this trial.
3. **Choose the next trial.** Review the current trial's metrics and the best trial so far. Consider the candidates you have already evaluated. You can tune a promising detector's core hyperparameters, revise the preprocessing steps, or try another detector family. Base each change on an observed characteristic of the series or a result from a previous trial.
4. **Handle failures.** If a tool returns a correctable error, use the error message to revise the candidate and resubmit it under the same task. Only successful evaluations count against the trial budget.

## Workspace and Experiment Artifacts

The experiment tools automatically manage all task artifacts, including those from each trial, under `.evotsad/` in the current workspace.

```text
.evotsad/
├── task.json                    # Configuration shared by all trials
├── profile.json                 # Training profile, observations, and baseline scores/errors
├── history.jsonl                # Rationale, metrics, and artifact paths for each successful trial
├── attempts.jsonl               # Local successful and failed attempts; not cross-task memory
└── experiments/                 # Per-trial directories
    └── exp-001/                 # Automatically assigned, unique trial ID
        ├── spec.json            # Configuration used to run this trial
        ├── model.zip            # Trained pipeline
        ├── scores.csv           # Evaluation scores
        ├── metrics.json         # Evaluation metrics
        └── manifest.json        # Artifact manifest
```

Use `get_history()` to compare successful candidates, `get_experiment(experiment_id)` for a full record, and `get_best_experiment()` for the current leader. The returned `experiment_dir` contains the trial artifacts. Baselines are reference scores, not successful trials, and cannot become `best_experiment` without a budgeted trial.

## Stop and Deliver

When the trial budget is exhausted the task ends automatically. If the goal is reached early or the user asks to stop, call `finish_task(reason)` before the final answer. A single trial completing is not task completion. After finish_task no further trials are allowed for that task.

Memory is written only once the entire task ends. During trials it is read-only; current attempts stay in the task log. The modes all, improvements, case_bank and contrastive have separate stores; none never reads or writes cross-task memory. Never use bash or file tools to inspect another mode's memory. Do not use task names or filenames as evidence of dataset properties.

Select the best evaluated trial on the primary metric and report its score and returned artifact paths. Report any memory or reflection warning honestly. Contrastive commentary is an unverified, conditional suggestion; computed differences and embedded before/after experiments are the evidence. Different-task scores are not directly comparable.
