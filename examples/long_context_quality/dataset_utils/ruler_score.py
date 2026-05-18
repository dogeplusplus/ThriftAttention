"""RULER scoring — port of NVIDIA/RULER/scripts/eval/synthetic/constants.py.

Two string-match variants matching the upstream metrics:

  string_match_part : mean(|{r in pred for r in ref}| / |ref|)     for CWE/FWE/QA
  string_match_all  : mean([all(r in pred) for r in ref])          for NIAH/VT

Both return a 0–100 score matching NVIDIA's reporting convention.
"""
from __future__ import annotations

from typing import Iterable, List


def _norm(s: str) -> str:
    return s.lower()


def string_match_part(pred: str, refs: Iterable[str]) -> float:
    """Recall: fraction of refs that appear (case-insensitive) in pred."""
    refs = list(refs)
    if not refs:
        return 0.0
    p = _norm(pred)
    return sum(1.0 for r in refs if _norm(r) in p) / len(refs)


def string_match_all(pred: str, refs: Iterable[str]) -> float:
    """All-or-nothing: 1.0 iff every ref is a substring of pred, else 0.0."""
    refs = list(refs)
    if not refs:
        return 0.0
    p = _norm(pred)
    return 1.0 if all(_norm(r) in p for r in refs) else 0.0


# Task → metric. Mirrors NVIDIA/RULER/scripts/eval/synthetic/constants.py.
TASK_METRIC = {
    "niah_single_1":   string_match_all,
    "niah_single_2":   string_match_all,
    "niah_single_3":   string_match_all,
    "niah_multikey_1": string_match_all,
    "niah_multikey_2": string_match_all,
    "niah_multikey_3": string_match_all,
    "niah_multivalue": string_match_all,
    "niah_multiquery": string_match_all,
    "vt":              string_match_all,
    "cwe":             string_match_part,
    "fwe":             string_match_part,
    "qa_1":            string_match_part,
    "qa_2":            string_match_part,
}


def score_sample(task: str, pred: str, refs: List[str]) -> float:
    """Score one prediction under the task's metric. Returns 0.0–1.0."""
    metric = TASK_METRIC.get(task)
    if metric is None:
        raise ValueError(f"unknown RULER task: {task}")
    return metric(pred, refs)
