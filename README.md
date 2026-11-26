# EvoTSAD

EvoTSAD 是一个面向时间序列异常检测的实验代理。它通过交互式对话帮助用户：

- 了解 LibraryTS 中可用的异常检测器和预处理器；
- 根据训练数据画像提出候选管线；
- 运行训练、评分和评估实验；
- 保存每次试验的配置、指标和模型产物；
- 从历史任务中检索相似数据条件下的实验经验，并在任务结束时按记忆模式保存新经验。

当前训练数据画像包含 17 个字段：5 个结构字段（训练行数、数值特征数、缺失比例、常数列比例、规则采样）和 12 个信号/事件字段。画像只读取训练 CSV；记忆相似度匹配使用这 17 个字段。画像不会直接替代模型选择或自动修改实验管线。

## 目录结构

```text
src/evotsad/
├── chat.py                 # 终端聊天入口
├── agent/                  # 代理循环
├── models/                 # 模型路由和响应模型
├── tools/experiments.py    # task_init、run_trial、search_memory 等工具
├── runtime/                # 任务状态、数据画像和运行时支持
├── evolution/              # 跨任务记忆、案例筛选和对比反思
└── prompts/                # 代理行为和工具使用说明

.evotsad/                   # 当前任务的任务状态和实验产物
.agent/                     # 会话、模型配置和跨任务记忆
```

## 环境准备

项目要求 Python 3.11 或更高版本，并需要可执行的 LibraryTS CLI。使用已有虚拟环境时，可以直接在仓库根目录执行：

```powershell
$env:PYTHONPATH = "src"
```

首次安装项目依赖可以使用：

```powershell
uv sync
```

在 `.agent/config.toml` 中选择默认模型和 provider，并在 `.agent/.env` 或系统环境变量中设置对应的 API key。不要把真实密钥提交到 Git。

## 启动

从 `code/EvoTSAD` 目录运行：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m evotsad.chat
```

也可以选择记忆模式：

```powershell
.\.venv\Scripts\python.exe -m evotsad.chat --memory-mode case_bank
```

支持的模式为：

- `all`：保存所有成功试验；
- `improvements`：保存首次成功和刷新最佳结果的试验；
- `case_bank`：保存代表性最佳、近优高效、结构多样和失败案例；
- `contrastive`：在案例基础上保存配置变化的对比证据；
- `none`：不读取、不写入跨任务记忆。

如果希望读取历史记忆但不更新它，增加：

```powershell
.\.venv\Scripts\python.exe -m evotsad.chat --memory-mode contrastive --no-update-memory
```

此开关只禁止长期记忆写入，当前任务的 `.evotsad/` 日志仍会正常保存。

## 批量会话

从仓库目录执行批量入口；每个 JSONL 行都是一个独立的新会话，行内 `tasks` 按顺序作为用户消息发送到与 `chat` 相同的代理循环。每个任务的 `prompt` 应包含聊天时会输入的完整要求（包括数据路径和试验预算），不支持 `/new`、`/image` 等交互式终端命令；如需结束任务以发布经验，也应在会话中提供相应的后续消息。会话之间顺序执行，先结束的会话所发布的记忆可供之后的会话读取。

```jsonl
{"memory_mode":"all","update_memory":true,"tasks":[{"prompt":"使用 train.csv 和 eval.csv 检测异常，最多运行 3 次试验。"},{"prompt":"请总结最佳模型并结束当前任务。"}]}
{"memory_mode":"case_bank","update_memory":false,"model":"provider/model","title":"复测","tasks":[{"prompt":"使用另一组训练及评估数据运行 2 次试验。"}]}
```

`memory_mode` 默认为 `all`，可选值与交互模式相同；`update_memory` 默认为 `true`，等价于是否设置 `--no-update-memory`。`model` 和 `title` 可选，省略时使用默认模型和会话标题。

```powershell
.\.venv\Scripts\python.exe -m evotsad.batch sessions.jsonl answers.jsonl
```

输出文件必须尚不存在。每行对应输入的一行，保留原有字段和任务顺序，并为每个任务增加 `response`（模型最终文本）。批量运行仍会在 `.agent/sessions/` 保留普通聊天会话，并在 `.evotsad/` 保存当前任务产物。遇到失败时，会在该任务写入 `error` 后停止；输出文件保留此前已完成的会话，但不会自动续跑。


## 任务使用流程

启动后直接输入任务要求。代理会在信息齐全后调用 `task_init`，需要提供：

- `train`：训练 CSV；
- `eval`：带 `label` 列的评估 CSV，且必须与训练文件不同；
- 可选的 `val`：验证 CSV；
- 目标、最大试验次数和可选主指标。

随后代理通常按以下流程工作：

1. 读取训练数据画像并检索初始历史经验；
2. 使用 `list_components` 查询检测器和预处理器；
3. 通过 `run_trial` 提交候选管线；
4. 根据指标、数据特征和已有试验结果调整下一轮管线；
5. 达到试验预算后自动结束，或调用 `finish_task` 提前结束。

一个管线由零个或多个预处理器加一个检测器组成，例如：

```json
[
  {"class": "StandardScaler"},
  {"class": "LSTMAD", "params": {"win_len": 96}}
]
```

`list_components` 应用于确认类名和参数；`run_trial` 会使用当前任务的训练、验证、评估路径和随机种子调用 LibraryTS。

## 输入和输出

输入文件使用 UTF-8 CSV。训练文件至少需要一个数值序列列；`timestamp` 和 `label` 会被识别为特殊列。评估文件必须包含完整的二元 `label` 列。

当前任务目录结构如下：

```text
.evotsad/
├── task.json              # 任务配置和画像快照
├── profile.json           # 17 字段画像、观察记录及五个基线结果
├── history.jsonl          # 成功试验记录
├── attempts.jsonl         # 成功和失败尝试记录
└── experiments/
    └── exp-001/
        ├── spec.json      # LibraryTS 实验配置
        ├── model.zip      # 训练后的模型
        ├── scores.csv     # 评估数据的异常分数
        ├── metrics.json   # 评估指标
        └── manifest.json  # 产物清单
```

试验结果中的 `spec_path` 和 `experiment_dir` 是查看实际产物的可靠路径。五个基线仅作参考，不占试验预算，也不参与最佳试验排名。不同任务的指标不能直接比较，应结合当前数据和任务目标解释历史经验。
