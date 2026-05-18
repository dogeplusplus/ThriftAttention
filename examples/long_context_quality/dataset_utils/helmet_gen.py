"""Small HELMET generators for the long-context mini examples.

This mirrors ``utils.ruler_gen``: it returns deterministic sample dicts under
an HF tokenizer, caches rendered prompts, and keeps task support small.

Supported HELMET tasks:
  * json_kv
  * kilt_popqa_3
  * narrativeqa

JSON KV and PopQA read the preprocessed HELMET JSONL files from
``HELMET_DATA_DIR`` or ``HELMET_REPO/data``. NarrativeQA streams from Hugging
Face datasets to avoid the full HELMET preprocessing path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)
HELMET_CACHE_DIR = Path(os.environ.get("HELMET_CACHE_DIR", Path.home() / ".cache" / "helmet_mini_data"))
HELMET_LENGTHS = [8192, 16384, 32768, 65536, 131072]

_JSONKV_K = {8192: 105, 16384: 220, 32768: 440, 65536: 900, 131072: 1800}
_POPQA_K = {8192: 50, 16384: 105, 32768: 220, 65536: 440, 131072: 1000}
TASK_NAMES = ["json_kv", "kilt_popqa_3", "narrativeqa"]
TASK_ALIASES = {"popqa": "kilt_popqa_3", "narrative_qa": "narrativeqa", "long_narrative_qa": "narrativeqa"}


def normalize_task(task: str) -> str:
    task = task.strip().lower()
    return TASK_ALIASES.get(task, task)


def _ntokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _tokenizer_id(tokenizer) -> str:
    return getattr(tokenizer, "name_or_path", None) or tokenizer.__class__.__name__


def _cache_path(cache_dir: Path, tokenizer_id: str, task: str, target_length: int, seed: int, num_samples: int) -> Path:
    tid = re.sub(r"[^A-Za-z0-9._-]", "_", tokenizer_id)
    return cache_dir / "samples" / tid / f"{task}_{target_length}_{seed}_{num_samples}.jsonl"


def _default_data_dir() -> Path:
    if os.environ.get("HELMET_DATA_DIR"):
        return Path(os.environ["HELMET_DATA_DIR"]).expanduser()
    return Path(os.environ.get("HELMET_REPO", Path.home() / "HELMET")).expanduser() / "data"


def _resolve_data_dir(data_dir: Optional[Path]) -> Path:
    base = Path(data_dir).expanduser() if data_dir else _default_data_dir()
    if (base / "data").is_dir() and not (base / "json_kv").exists() and not (base / "kilt").exists():
        base = base / "data"
    return base


def _data_file(data_dir: Path, relative: str) -> Path:
    path = data_dir / relative
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Missing HELMET data file: {path}\n"
        "Set --helmet-data-dir or HELMET_DATA_DIR to a HELMET data directory. "
        "This mini script only needs the requested json_kv/*.jsonl and kilt/popqa*.jsonl files; "
        "NarrativeQA is loaded from HF datasets."
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_rows(rows: List[Dict[str, Any]], num_samples: int, seed: int) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        return []
    if len(rows) <= num_samples:
        return list(rows)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[i] for i in indices[:num_samples]]


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, list):
                out.extend(str(x) for x in item)
            else:
                out.append(str(item))
        return out
    return [str(value)]


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _target_prompt_budget(target_length: int, max_gen: int) -> int:
    return max(1, target_length - max_gen - 100)


def _trim_context(tokenizer, context: str, keep_tokens: int) -> str:
    if keep_tokens <= 0:
        return ""
    ids = tokenizer.encode(context, add_special_tokens=False)
    if len(ids) <= keep_tokens:
        return context
    if getattr(tokenizer, "is_fast", False):
        try:
            enc = tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
            offsets = enc.get("offset_mapping") or []
            if keep_tokens <= len(offsets):
                return context[: offsets[keep_tokens - 1][1]]
        except Exception:
            pass
    return tokenizer.decode(ids[:keep_tokens], skip_special_tokens=False)


def _fit_context(tokenizer, context: str, render_input, answer_prefix: str, max_prompt_tokens: int) -> str:
    input_text = render_input(context)
    if _ntokens(tokenizer, input_text + answer_prefix) <= max_prompt_tokens:
        return input_text

    fixed_tokens = _ntokens(tokenizer, render_input("") + answer_prefix)
    keep_tokens = max_prompt_tokens - fixed_tokens
    trimmed = context
    for _ in range(4):
        trimmed = _trim_context(tokenizer, trimmed, keep_tokens)
        input_text = render_input(trimmed)
        actual = _ntokens(tokenizer, input_text + answer_prefix)
        if actual <= max_prompt_tokens:
            return input_text
        keep_tokens = max(0, keep_tokens - (actual - max_prompt_tokens) - 8)
    return input_text


def _build_spec(task: str, target_length: int) -> Dict[str, Any]:
    task = normalize_task(task)
    if target_length not in HELMET_LENGTHS:
        raise ValueError(f"unsupported HELMET length {target_length}; expected one of {HELMET_LENGTHS}")
    if task == "json_kv":
        return {
            "task": task,
            "category": "recall",
            "test_file": f"json_kv/test_k{_JSONKV_K[target_length]}_dep6.jsonl",
            "gen_max": 100,
            "shots": 2,
            "metric": "substring_exact_match",
            "stop_newline": False,
        }
    if task == "kilt_popqa_3":
        k = _POPQA_K[target_length]
        return {
            "task": task,
            "category": "rag",
            "test_file": f"kilt/popqa_test_1000_k{k}_dep6.jsonl",
            "demo_file": "kilt/popqa_test_1000_k3_dep6.jsonl",
            "gen_max": 20,
            "shots": 2,
            "metric": "substring_exact_match",
            "stop_newline": True,
            "popularity_threshold": 3.0,
        }
    if task == "narrativeqa":
        return {
            "task": task,
            "category": "longqa",
            "gen_max": 100,
            "shots": 2,
            "metric": "rougeL_f1",
            "stop_newline": False,
        }
    raise ValueError(f"unknown HELMET task: {task}; supported tasks are {TASK_NAMES}")


def _json_kv_samples(tokenizer, spec: Dict[str, Any], target_length: int, num_samples: int, seed: int, data_dir: Path) -> List[Dict[str, Any]]:
    rows = _select_rows(_read_jsonl(_data_file(data_dir, spec["test_file"])), num_samples, seed)
    max_prompt_tokens = _target_prompt_budget(target_length, spec["gen_max"])
    answer_prefix = "\nCorresponding value:"
    samples: List[Dict[str, Any]] = []

    for row in rows:
        demos = []
        for item in row.get("demos", [])[: spec["shots"]]:
            if isinstance(item, dict):
                key = item.get("key", item.get("question", ""))
                value = item.get("value", item.get("answer", ""))
            else:
                key, value = item[0], item[1]
            value = str(value)
            if not value.startswith(" "):
                value = " " + value
            demos.append(f"Key: {key}\nCorresponding value:{value}")
        demos_text = "\n\n".join(demos) + ("\n\n" if demos else "")
        question = str(row.get("question", row.get("key", "")))
        context = str(row.get("context", ""))

        def render(ctx: str) -> str:
            return (
                f"{ctx}\n\n"
                "Extract the value corresponding to the specified key in the JSON object below.\n\n"
                f"{demos_text}Key: {question}"
            )

        input_text = _fit_context(tokenizer, context, render, answer_prefix, max_prompt_tokens)
        prompt_tokens = _ntokens(tokenizer, input_text + answer_prefix)
        samples.append(
            {
                "task": spec["task"],
                "category": spec["category"],
                "target_length": target_length,
                "length": prompt_tokens + spec["gen_max"],
                "input": input_text,
                "answer_prefix": answer_prefix,
                "outputs": _as_list(row.get("answer", row.get("value"))),
                "max_gen": spec["gen_max"],
                "metric": spec["metric"],
                "stop_newline": spec["stop_newline"],
                "extra": {"num_kvs": row.get("num_kvs")},
            }
        )
    return samples


def _record_key(row: Dict[str, Any]) -> Tuple[str, str]:
    for key in ("id", "qid", "question", "query"):
        if key in row:
            return key, str(row[key])
    return "row", json.dumps(row, sort_keys=True)[:200]


def _dedupe_by_key(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = _record_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _passes_popularity(row: Dict[str, Any], threshold: Optional[float]) -> bool:
    if threshold is None or "s_pop" not in row:
        return True
    try:
        return math.log10(float(row["s_pop"])) < threshold
    except (TypeError, ValueError):
        return True


def _passage_text(ctxs: Iterable[Dict[str, Any]]) -> str:
    passages = []
    for ctx in ctxs:
        title = str(ctx.get("title", "")).strip()
        text = str(ctx.get("text", ctx.get("contents", ""))).strip()
        if title:
            passages.append(f"Document (Title: {title}): {text}")
        else:
            passages.append(f"Document: {text}")
    return "\n\n".join(passages)


def _popqa_samples(tokenizer, spec: Dict[str, Any], target_length: int, num_samples: int, seed: int, data_dir: Path) -> List[Dict[str, Any]]:
    threshold = spec.get("popularity_threshold")
    rows = [row for row in _read_jsonl(_data_file(data_dir, spec["test_file"])) if _passes_popularity(row, threshold)]
    rows = _select_rows(rows, num_samples, seed)
    demo_rows = _read_jsonl(_data_file(data_dir, spec["demo_file"]))
    demo_rows = _dedupe_by_key(row for row in demo_rows if _passes_popularity(row, threshold))

    max_prompt_tokens = _target_prompt_budget(target_length, spec["gen_max"])
    answer_prefix = "\nAnswer:"
    samples: List[Dict[str, Any]] = []

    for row in rows:
        key_name, key_value = _record_key(row)
        candidates = [d for d in demo_rows if str(d.get(key_name, "")) != key_value]
        rng = random.Random(_stable_seed(spec["task"], key_value))
        rng.shuffle(candidates)
        demo_chunks = []
        for demo in candidates[: spec["shots"]]:
            demo_answer = _as_list(demo.get("answers", demo.get("answer")))
            demo_chunks.append(
                "{documents}\n\nQuestion: {question}\nAnswer: {answer}".format(
                    documents=_passage_text(demo.get("ctxs", [])),
                    question=demo.get("question", ""),
                    answer=demo_answer[0] if demo_answer else "",
                )
            )
        demos_text = "\n\n".join(demo_chunks) + ("\n\n" if demo_chunks else "")
        question = str(row.get("question", row.get("query", "")))
        context = _passage_text(row.get("ctxs", []))

        def render(ctx: str) -> str:
            return (
                "Use the given documents to write a concise and short answer to the question.\n"
                "Write your answer in the following format:\n"
                "Answer: [answer]\n\n"
                f"{demos_text}{ctx}\n\nQuestion: {question}"
            )

        input_text = _fit_context(tokenizer, context, render, answer_prefix, max_prompt_tokens)
        prompt_tokens = _ntokens(tokenizer, input_text + answer_prefix)
        samples.append(
            {
                "task": spec["task"],
                "category": spec["category"],
                "target_length": target_length,
                "length": prompt_tokens + spec["gen_max"],
                "input": input_text,
                "answer_prefix": answer_prefix,
                "outputs": _as_list(row.get("answers", row.get("answer"))),
                "max_gen": spec["gen_max"],
                "metric": spec["metric"],
                "stop_newline": spec["stop_newline"],
                "extra": {key_name: key_value},
            }
        )
    return samples


def _nqa_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("text", "")
    if isinstance(value, list):
        return " ".join(_nqa_text(v) for v in value)
    return "" if value is None else str(value)


def _nqa_answers(value: Any) -> List[str]:
    if isinstance(value, dict):
        return _as_list(value.get("text", value.get("answers", [])))
    if isinstance(value, list):
        return [text for text in (_nqa_text(v) for v in value) if text]
    return _as_list(value)


def _nqa_fields(row: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    context = _nqa_text(row.get("document", row.get("context", "")))
    question = _nqa_text(row.get("question", ""))
    answers = _nqa_answers(row.get("answers", row.get("answer", [])))
    return context, question, answers


def _load_narrativeqa_demo_text(shots: int, seed: int) -> str:
    if shots <= 0:
        return ""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("NarrativeQA needs `pip install datasets`.") from exc

    train = load_dataset("narrativeqa", split="train", streaming=True)
    try:
        train = train.shuffle(seed=seed, buffer_size=512)
    except Exception:
        pass

    demos = []
    for row in train:
        _, question, answers = _nqa_fields(row)
        if question and answers:
            demos.append(f"Question: {question}\nAnswer: {answers[0]}")
        if len(demos) >= shots:
            break
    if not demos:
        return ""
    return "For example:\n\n" + "\n\n".join(demos) + "\n\nNow, use the following story to answer the question:\n\n"


def _render_chat_or_plain(tokenizer, user_text: str, system_template: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return user_text + "\n" + system_template


def _narrativeqa_samples(tokenizer, spec: Dict[str, Any], target_length: int, num_samples: int, seed: int) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("NarrativeQA needs `pip install datasets`.") from exc

    data = load_dataset("narrativeqa", split="test", streaming=True)
    try:
        data = data.shuffle(seed=seed, buffer_size=1024)
    except Exception:
        pass

    demo_text = _load_narrativeqa_demo_text(spec["shots"], seed)
    max_prompt_tokens = _target_prompt_budget(target_length, spec["gen_max"])
    samples: List[Dict[str, Any]] = []
    max_scan = max(2000, num_samples * 200)

    for scanned, row in enumerate(data, start=1):
        context, question, answers = _nqa_fields(row)
        if not context or not question or not answers:
            continue

        def render(ctx: str) -> str:
            user_text = (
                "You are given a story, which can be either a novel or a movie script, and a question.\n"
                "Answer the question as concisely as you can, using a single phrase if possible.\n\n"
                f"{demo_text}{ctx}\n\nQuestion: {question}"
            )
            return _render_chat_or_plain(tokenizer, user_text, "Answer:")

        input_text = _fit_context(tokenizer, context, render, "", max_prompt_tokens)
        prompt_tokens = _ntokens(tokenizer, input_text)
        samples.append(
            {
                "task": spec["task"],
                "category": spec["category"],
                "target_length": target_length,
                "length": prompt_tokens + spec["gen_max"],
                "input": input_text,
                "answer_prefix": "",
                "outputs": answers,
                "max_gen": spec["gen_max"],
                "metric": spec["metric"],
                "stop_newline": spec["stop_newline"],
                "extra": {},
            }
        )
        if len(samples) >= num_samples or scanned >= max_scan:
            break

    if len(samples) < num_samples:
        LOGGER.warning("NarrativeQA produced %d/%d samples", len(samples), num_samples)
    return samples


def generate_samples(
    tokenizer,
    task: str,
    target_length: int,
    num_samples: int = 100,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Generate or load HELMET samples for one (task, length).

    Returns sample dicts shaped like ``ruler_gen.generate_samples``:
        {task, input, answer_prefix, outputs, max_gen, length, target_length}
    """
    task = normalize_task(task)
    spec = _build_spec(task, target_length)
    cache_dir = Path(cache_dir) if cache_dir else HELMET_CACHE_DIR
    cache_file = _cache_path(cache_dir, _tokenizer_id(tokenizer), task, target_length, seed, num_samples)
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    LOGGER.info("Generating HELMET %s @ %d tokens (n=%d, seed=%d)", task, target_length, num_samples, seed)
    if task == "json_kv":
        samples = _json_kv_samples(tokenizer, spec, target_length, num_samples, seed, _resolve_data_dir(data_dir))
    elif task == "kilt_popqa_3":
        samples = _popqa_samples(tokenizer, spec, target_length, num_samples, seed, _resolve_data_dir(data_dir))
    elif task == "narrativeqa":
        samples = _narrativeqa_samples(tokenizer, spec, target_length, num_samples, seed)
    else:
        raise ValueError(f"unknown HELMET task: {task}")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
    return samples
