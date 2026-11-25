"""Mode-isolated memory, published once per completed task."""

import json
import math
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from ..runtime.file_lock import locked
from .case_bank import select_cases
from .contrastive import build_contrasts

MEMORY_MODES = ("all", "improvements", "none", "case_bank", "contrastive")


def _tokens(text):
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.casefold()))


def _profile_similarity(left, right):
    scores = []
    for key in ("training_rows", "numeric_features", "missing_ratio", "constant_column_ratio", "regular_sampling",
                "rolling_location_max", "rolling_location_median",
                "SB_BinaryStats_diff_longstretch0", "kpss_level_auto_statistic",
                "train_eval_median_shift", "excess_kurtosis", "permutation_entropy",
                "number_peaks", "B01", "A03", "A04", "C01"):
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            continue
        if key in {"A04", "training_rows", "numeric_features"}:
            scores.append(1 / (1 + abs(math.log1p(a)-math.log1p(b))))
        elif key == "regular_sampling":
            scores.append(float(a == b))
        else:
            scores.append(max(0, 1-abs(a-b)/(2 if key in {"excess_kurtosis", "C01"} else 1)))
    return sum(scores)/len(scores) if scores else None


class ExperienceMemory:
    def __init__(self, workspace: Path, mode: str = "all") -> None:
        if mode not in MEMORY_MODES:
            raise ValueError(f"Unknown memory mode: {mode!r}")
        self.mode = mode
        self.path = workspace / ".agent" / "memory" / mode / "memory.jsonl"

    def finalize(self, task: dict, attempts: list[dict], reflector=None) -> dict:
        if self.mode == "none":
            return {"mode": self.mode, "saved": False, "cases": 0, "contrasts": 0, "warnings": []}
        if task.get("status") != "ended":
            raise ValueError("Memory can only be updated after the whole task ends.")
        warnings = []
        if self.mode in {"case_bank", "contrastive"}:
            cases = select_cases(task, attempts)
        else:
            cases, best = [], None
            for attempt in attempts:
                if attempt["status"] != "completed":
                    continue
                score = attempt["metrics"][task["primary_metric"]]
                if self.mode == "all" or best is None or score > best:
                    cases.append({"task_description": task["task_description"], "profile": task["profile"],
                                  "roles": [self.mode], "pipeline": deepcopy(attempt["pipeline"]),
                                  "evidence": [deepcopy(attempt)]})
                best = score if best is None else max(best, score)
        contrasts = []
        if self.mode == "contrastive":
            contrasts, warnings = build_contrasts(task, attempts, reflector)
            # Link observations by actual configuration changes, never by task identity.
            prior = []
            if self.path.exists() and self.path.stat().st_size:
                with self.path.open(encoding="utf-8") as stream:
                    with locked(stream, shared=True):
                        prior = [json.loads(line) for line in stream if line.strip()]
            for entry in contrasts:
                related = []
                for old in reversed(prior):
                    if old.get("mode") != self.mode or old.get("primary_metric") != task["primary_metric"]:
                        continue
                    for contrast in old.get("contrasts", []):
                        if (contrast["comparison_type"] == entry["comparison_type"]
                                and contrast["config_diff"] == entry["config_diff"]
                                and contrast["before"]["pipeline"] == entry["before"]["pipeline"]
                                and contrast["after"]["pipeline"] == entry["after"]["pipeline"]
                                and contrast["before"].get("software") == entry["before"].get("software")):
                            # Avoid recursive growth: embed evidence but not previous links.
                            related.append({k: deepcopy(v) for k, v in contrast.items() if k != "related_observations"})
                        if len(related) == 2:
                            break
                    if len(related) == 2:
                        break
                entry["related_observations"] = related
                delta = entry.get("metric_delta")
                tolerance = 0.01 if task["primary_metric"] == "VUS-PR" else 0
                if delta is not None and abs(delta) > tolerance and any(
                        r.get("metric_delta") is not None and abs(r["metric_delta"]) > tolerance
                        and delta*r["metric_delta"] < 0 for r in related):
                    entry["status"] = "conflicting_observations"
                    entry["limitations"].append("Related tasks have opposite observations; data conditions may differ.")
        bundle = {"schema_version": 1, "mode": self.mode,
                  "task_description": task["task_description"], "profile": task["profile"],
                  "primary_metric": task["primary_metric"], "metric_params": task["metrics"],
                  "task_settings": {k: deepcopy(task[k]) for k in ("objective", "max_trials", "seed", "data", "software")},
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "cases": cases, "contrasts": contrasts, "warnings": warnings}
        content = json.dumps(bundle, ensure_ascii=False, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as stream:
            with locked(stream):
                stream.write(content + "\n")
                stream.flush()
        return {"mode": self.mode, "saved": True, "path": str(self.path),
                "cases": len(cases), "contrasts": len(contrasts), "warnings": warnings}

    def search(self, query: str = "", limit: int = 5, *, profile=None, primary_metric=None,
               max_chars: int = 16000) -> list[dict]:
        if not isinstance(query, str) or type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("query must be text; limit must be an integer between 1 and 50.")
        if self.mode == "none" or not self.path.exists() or self.path.stat().st_size == 0:
            return []
        with self.path.open("r", encoding="utf-8") as stream:
            with locked(stream, shared=True):
                bundles = [json.loads(line) for line in stream if line.strip()]
        ranked = []
        query_tokens = _tokens(query)
        for order, bundle in enumerate(bundles):
            if bundle.get("mode") != self.mode:
                continue
            if primary_metric and bundle["primary_metric"] != primary_metric:
                continue
            similarity = _profile_similarity(profile or {}, bundle.get("profile") or {})
            for kind in ("cases", "contrasts"):
                for entry in bundle[kind]:
                    # Paths, task names and identifiers never enter the similarity score.
                    searchable = {key: entry[key] for key in ("pipeline", "observed_result", "reflection", "roles", "config_diff") if key in entry}
                    tokens = _tokens(json.dumps(searchable, ensure_ascii=False))
                    if query_tokens and not query_tokens & tokens:
                        continue
                    lexical = len(query_tokens & tokens)/len(query_tokens | tokens) if query_tokens else 0
                    score = lexical if similarity is None else 0.8*similarity+0.2*lexical
                    item = {"mode": self.mode, "type": kind, "primary_metric": bundle["primary_metric"],
                            "match": {"profile_similarity": similarity, "text_similarity": lexical}, **deepcopy(entry)}
                    ranked.append((score, order, kind, item))
        ranked.sort(key=lambda v: (v[0], v[1], v[2] == "contrasts"), reverse=True)
        selected, counts, total = [], {}, 0
        contrasts = failures = 0
        for _, order, kind, item in ranked:
            if counts.get(order, 0) >= 2:
                continue
            if kind == "contrasts" and contrasts >= 2:
                continue
            failure = "failure" in item.get("roles", [])
            if failure and failures >= limit//3:
                continue
            size = len(json.dumps(item, ensure_ascii=False))
            if total + size > max_chars:
                continue  # Preserve complete task text and evidence, never truncate them.
            selected.append(item)
            total += size
            counts[order] = counts.get(order, 0)+1
            contrasts += kind == "contrasts"
            failures += failure
            if len(selected) == limit:
                break
        return selected
