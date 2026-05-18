"""Scoring helpers for the mini HELMET subset."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List, Optional

PRIMARY_METRIC = {
    "json_kv": "substring_exact_match",
    "kilt_popqa_3": "substring_exact_match",
    "narrativeqa": "rougeL_f1",
}


def _normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc(text.lower())).split())


def _flatten_refs(refs) -> List[str]:
    if isinstance(refs, str):
        return [refs]
    out: List[str] = []
    for ref in refs:
        if isinstance(ref, list):
            out.extend(str(x) for x in ref)
        else:
            out.append(str(ref))
    return out


def _exact_match(prediction: str, ref: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ref))


def _substring_exact_match(prediction: str, ref: str) -> float:
    return float(_normalize_answer(ref) in _normalize_answer(prediction))


def _f1(prediction: str, ref: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _rouge_l_f1(prediction: str, refs) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("NarrativeQA scoring needs `pip install rouge_score`.") from exc

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return max(scorer.score(target=ref, prediction=prediction)["rougeL"].fmeasure for ref in _flatten_refs(refs))


def _parse_output(output: str, prefix: str = "Answer:") -> Optional[str]:
    patterns = [
        re.compile(f"(?:{re.escape(prefix)})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pattern in patterns:
        match = pattern.search(output)
        if match is not None:
            parsed = match.group(1).strip()
            return re.sub(f"^{re.escape(prefix)}", "", parsed, flags=re.IGNORECASE).strip()
    return None


def _max_over(fn, prediction: str, refs: Iterable[str]) -> float:
    refs = list(refs)
    if not refs:
        return 0.0
    return max(fn(prediction, ref) for ref in refs)


def _string_metric(metric: str, prediction: str, refs) -> float:
    flat = _flatten_refs(refs)
    if metric == "exact_match":
        return _max_over(_exact_match, prediction, flat)
    if metric == "substring_exact_match":
        return _max_over(_substring_exact_match, prediction, flat)
    if metric == "f1":
        return _max_over(_f1, prediction, flat)
    if metric == "rougeL_f1":
        return _rouge_l_f1(prediction, flat)
    raise ValueError(f"unknown HELMET metric: {metric}")


def score_sample(task: str, prediction: str, sample) -> float:
    """Score one HELMET mini prediction. Returns 0.0-1.0."""
    metric = PRIMARY_METRIC.get(task)
    if metric is None:
        raise ValueError(f"unknown HELMET task: {task}")
    refs = sample.get("outputs")
    if refs is None or (isinstance(refs, (list, tuple)) and len(refs) == 0):
        return 0.0
    raw = _string_metric(metric, prediction, refs)
    prefix = "Corresponding value:" if task == "json_kv" else "Answer:"
    parsed = _parse_output(prediction, prefix)
    if parsed is None:
        return raw
    return max(raw, _string_metric(metric, parsed, refs))
