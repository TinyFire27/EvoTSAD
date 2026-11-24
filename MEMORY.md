# 跨任务记忆模式

启动聊天时用 `--memory-mode` 选择本次进程的模式：

```bash
python -m evotsad.chat --memory-mode all
python -m evotsad.chat --memory-mode improvements
python -m evotsad.chat --memory-mode none
```

- `all`（默认）：记录每次成功评估的试验，读取 `.agent/experience.jsonl`，兼容已有记忆。
- `improvements`：仅记录当前任务首次成功试验及主指标严格超过此前最佳值的试验；读取 `.agent/experience-improvements.jsonl`。主指标沿用现有的越大越好规则，相等不记录。
- `none`：不读取或写入跨任务记忆。当前任务的实验历史仍正常保存。

模式在启动时确定，对该进程中的所有会话生效。不同模式不会互读；切换模式需要重新启动。记忆检索使用文本包含匹配，历史结果仅供参考，应在新数据上重新评估。
