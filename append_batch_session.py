"""Append one batch session built from the same prompt template for each dataset."""

import json
from pathlib import Path


# Edit these settings before running the script.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "sessions.jsonl"
MEMORY_MODE = "all"
UPDATE_MEMORY = True
MAX_TRIALS = 3
DATASETS = [
    {"train": "data/example/train.csv", "eval": "data/example/eval.csv"},
    # {"train": "data/other/train.csv", "val": "data/other/val.csv", "eval": "data/other/eval.csv"},
]

PROMPT_TEMPLATE = (
    "请使用训练集 {train}、评估集 {eval}{validation}进行时间序列异常检测，"
    "最多进行 {max_trials} 次正式试验。请比较候选模型的效果，选出最佳模型，"
    "任务结束时调用 finish_task 并报告结果。"
)


def main() -> None:
    tasks = []
    for dataset in DATASETS:
        validation = f"、验证集 {dataset['val']}" if dataset.get("val") else ""
        tasks.append({"prompt": PROMPT_TEMPLATE.format(
            train=dataset["train"],
            eval=dataset["eval"],
            validation=validation,
            max_trials=MAX_TRIALS,
        )})

    session = {"memory_mode": MEMORY_MODE, "update_memory": UPDATE_MEMORY, "tasks": tasks}
    with CONFIG_PATH.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(session, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
