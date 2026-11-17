---
name: libraryts-cli
description: Develop and evaluate time-series anomaly detectors from local CSV data. This skill explains how to use the LibraryTS command-line tool (`libraryts`) to choose detectors and preprocessors, configure an experiment, train and save a pipeline, score new data, and evaluate the results when labels are available.
---

# LibraryTS CLI User Guide

## Purpose

This skill is the companion guide for the `libraryts` command-line tool. Use it to develop time-series anomaly detectors from local CSV files supplied by the user. You can discover components, assemble a detection pipeline, train and save it, score data with the saved pipeline, and compute metrics when the evaluation CSV contains labels. Run `libraryts --help` to verify that the CLI is available.

## Available Commands

| Command | Purpose |
|---|---|
| `libraryts ls` | Discover available detectors and preprocessors. |
| `libraryts train` | Train and save a detector or pipeline. |
| `libraryts predict` | Load a saved detector or pipeline and generate anomaly scores. |
| `libraryts evaluate` | Compute metrics from saved scores and the corresponding labels. |
| `libraryts run` | Train, score, and evaluate in one command, saving the outputs of all three stages. |

## Minimal Workflow for One Experiment

**The experiment spec is the entry point for an experiment.** Create a JSON file that specifies the data, ordered pipeline, parameters, and metrics for the experiment. Pass the file path to the CLI with `--spec`.

1. **Discover components.** Run `libraryts ls algorithms --task anomaly_detection` to list detectors. Add `--name CLASS` with the exact class name to see its parameters, defaults, and recommended preprocessing.
2. **Write the spec.** Create `exp-01.json` using the experiment spec reference. A complete `run` needs a training CSV, a separate labeled CSV for scoring and evaluation, the ordered pipeline, and at least one metric. Put any preprocessing steps before the detector. Use the data splits and labels allowed by the task.
3. **Run and inspect the results.** Execute `libraryts run --spec exp-01.json --output-dir outputs/exp-01` from the experiment workspace. Choose an output directory that does not exist yet. On success, read the metrics and artifact paths from the JSON on stdout. Use `libraryts train` if the task only requires a fitted detector or pipeline.

## Reference Index

| When to read | Reference |
|---|---|
| Before invoking a command, read its syntax, options, and error-handling rules. | [Commands](references/commands.md) |
| Before writing or editing a spec, read its fields, CSV requirements, and path rules. | [Experiment Spec](references/experiment-spec.md) |
| Before choosing an output directory or reusing an artifact, read the file layout and directory handoff rules. | [Files and Artifacts](references/artifacts.md) |
