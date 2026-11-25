"""Select representative, self-contained cases from one completed task."""

import json
from copy import deepcopy
from statistics import mean, stdev


def config_text(pipeline: list[dict]) -> str:
    normalized = [{"class": s["class"], "params": s.get("params", {})} for s in pipeline]
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, allow_nan=False)


def select_cases(task: dict, attempts: list[dict], tolerance: float = 0.01) -> list[dict]:
    metric = task["primary_metric"]
    groups = {}
    for attempt in attempts:
        if attempt["status"] == "completed":
            groups.setdefault(config_text(attempt["pipeline"]), []).append(attempt)
    candidates = []
    for key, evidence in groups.items():
        # Repeated execution of a seed is not an independent replicate.
        by_seed = {}
        for item in evidence:
            by_seed.setdefault(item["seed"], []).append(item["metrics"][metric])
        scores = [mean(values) for values in by_seed.values()]
        candidates.append({
            "task_description": task["task_description"], "profile": task["profile"],
            "pipeline": deepcopy(evidence[0]["pipeline"]), "roles": [],
            "score_summary": {"mean": mean(scores), "std": stdev(scores) if len(scores) > 1 else None,
                              "independent_seeds": len(scores), "observations": len(evidence)},
            "elapsed_seconds": mean(item["elapsed_seconds"] for item in evidence),
            "evidence": deepcopy(evidence), "_key": key,
        })
    candidates.sort(key=lambda c: (-c["score_summary"]["mean"], c["elapsed_seconds"], c["_key"]))
    chosen = []

    def add(case, role):
        if case not in chosen:
            chosen.append(case)
        case["roles"].append(role)

    if candidates:
        best = candidates[0]
        add(best, "best")
        # The absolute tolerance is defined only for VUS-PR in this first version.
        efficient = [c for c in candidates[1:] if metric == "VUS-PR"
                     and best["score_summary"]["mean"] - c["score_summary"]["mean"] <= tolerance
                     and c["elapsed_seconds"] < best["elapsed_seconds"]]
        if efficient:
            add(min(efficient, key=lambda c: (c["elapsed_seconds"], c["_key"])), "efficient")
        classes = {c["pipeline"][-1]["class"] for c in chosen}
        diverse = next((c for c in candidates if c["pipeline"][-1]["class"] not in classes), None)
        if diverse is None:
            structures = {tuple(s["class"] for s in c["pipeline"][:-1]) for c in chosen}
            diverse = next((c for c in candidates if tuple(s["class"] for s in c["pipeline"][:-1]) not in structures), None)
        if diverse:
            add(diverse, "diverse")
    failures = [a for a in attempts if a["status"] == "execution_failed"
                and a.get("failure_category") in {"configuration", "resource"}]
    if failures:
        failures.sort(key=lambda a: (a["failure_category"] != "configuration", a["created_at"]))
        failure = failures[0]
        chosen.append({"task_description": task["task_description"], "profile": task["profile"],
                       "pipeline": deepcopy(failure["pipeline"]), "roles": ["failure"],
                       "evidence": [deepcopy(failure)], "failure_category": failure["failure_category"]})
    for case in chosen:
        case.pop("_key", None)
    return chosen
