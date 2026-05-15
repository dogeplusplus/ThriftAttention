#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any


EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from common import (  # noqa: E402
    collect_environment,
    make_output_dir,
    markdown_table,
    parse_int_list,
    parse_str_list,
    set_seed,
    sync_cuda,
    thrift_acceleration_status,
    write_json,
    write_jsonl,
)


DEFAULT_MODEL = "Qwen/Qwen3-8B"
METHODS = {"fp16": "fp16", "flash": "fp16", "fp4": "fp4", "thrift": "thrift"}
TASK_ALIASES = {"needle": "niah_single_1", "variable_tracking": "vt", "common_words": "cwe"}
DEFAULT_RULER_DIR = Path(os.environ.get("RULER_EXPERIMENTS_DIR", "/workspace/nvfp4-experiments/experiments"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini RULER generation benchmark with explicit prefill/decode timing.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lengths", default="4096,8192")
    parser.add_argument("--tasks", default="niah_single_1,vt,cwe")
    parser.add_argument("--methods", default="fp16,fp4,thrift")
    parser.add_argument("--fractions", default="0.05")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 uses each RULER task's default generation length.")
    parser.add_argument("--ruler-dir", type=Path, default=DEFAULT_RULER_DIR, help="Directory containing ruler_gen.py and ruler_score.py.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="RULER sample cache directory.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output directory for metrics.jsonl and summary.md.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def method_runs(methods: str, fractions: str) -> list[dict[str, Any]]:
    frac_values = [float(item) for item in fractions.replace(" ", ",").split(",") if item.strip()]
    if not frac_values:
        raise SystemExit("--fractions must contain at least one value")
    out = []
    for raw in parse_str_list(methods):
        method = METHODS.get(raw.lower())
        if method is None:
            raise SystemExit(f"unknown method {raw!r}; choose fp16, fp4, thrift")
        if method == "thrift":
            for frac in frac_values:
                if not 0.0 <= frac <= 1.0:
                    raise SystemExit("--fractions must be in [0, 1]")
                out.append({"method": method, "label": f"thrift_{frac * 100:g}pct".replace(".", "p"), "fraction": frac})
        else:
            out.append({"method": method, "label": method, "fraction": None})
    if not out:
        raise SystemExit("--methods must contain at least one method")
    return out


def load_model(model_id: str, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return model, tokenizer


def configure(model: Any, method: str, fraction: float | None, device: str) -> tuple[bool, str, Any | None]:
    import thriftattention as ta

    try:
        ta.unpatch_model(model, backend="hf")
    except Exception:
        pass
    if method == "fp16":
        return True, "standard Transformers attention", None
    ready, note = thrift_acceleration_status(device)
    if not ready:
        return False, note, None
    try:
        from thriftattention.config import AttentionConfig

        mode = "fp4" if method == "fp4" else "thrift"
        config = AttentionConfig(mode=mode, fp16_fraction=0.0 if fraction is None else fraction, backend="hf", patch_generation=False)
        ta.patch_model(model, backend="hf", mode=mode, fp16_fraction=config.fp16_fraction, patch_generation=False)
    except Exception as exc:
        return False, str(exc), None
    return True, f"patched {method}" if method == "fp4" else f"patched thrift fraction={fraction:g}", config


def set_model_attention_config(model: Any, config: Any) -> None:
    from thriftattention.integrations import transformers as hf

    attr = getattr(hf, "_CONFIG_ATTR", "_thriftattention_config")
    modules = getattr(model, "modules", None)
    for module in modules() if callable(modules) else [model]:
        if hasattr(module, attr):
            setattr(module, attr, config)


def nvfp4_top_k_for_prompt(config: Any, prompt_len: int) -> Any:
    if config is None or getattr(config, "mode", None) != "thrift" or getattr(config, "top_k", None) is not None:
        return config
    from thriftattention.selection import resolve_top_k

    blocks = max(prompt_len // 64, 1)
    top_k = resolve_top_k(blocks, causal=True, fraction=config.fp16_fraction)
    return replace(config, top_k=top_k)


def load_ruler_modules(ruler_dir: Path) -> tuple[Any, Any]:
    if not (ruler_dir / "ruler_gen.py").exists() or not (ruler_dir / "ruler_score.py").exists():
        raise SystemExit(f"Could not find ruler_gen.py and ruler_score.py in {ruler_dir}")
    sys.path.insert(0, str(ruler_dir))
    import ruler_gen
    import ruler_score

    return ruler_gen, ruler_score


def normalise_tasks(tasks: list[str], ruler_gen: Any) -> list[str]:
    out = [TASK_ALIASES.get(task, task) for task in tasks]
    valid = set(getattr(ruler_gen, "TASK_NAMES", getattr(ruler_gen, "TASK_SPECS", {}).keys()))
    unknown = sorted(set(out) - valid)
    if unknown:
        raise SystemExit(f"unknown RULER task(s): {', '.join(unknown)}")
    return out


def generate_timed(model: Any, tokenizer: Any, sample: dict[str, Any], cache_config: Any | None, args: argparse.Namespace) -> dict[str, Any]:
    import time
    import torch

    prompt = sample["input"] + sample["answer_prefix"]
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_new_tokens = int(args.max_new_tokens or sample["max_gen"])
    prompt_len = len(input_ids)
    pad = (-prompt_len) % 64 if cache_config is not None else 0
    encoded = torch.tensor([input_ids + [0] * pad], dtype=torch.long, device=args.device)
    cache_position = torch.arange(encoded.shape[1], device=args.device, dtype=torch.long)
    past = None
    cache_ctx = nullcontext()
    active_config = nvfp4_top_k_for_prompt(cache_config, prompt_len)
    if active_config is not cache_config:
        set_model_attention_config(model, active_config)
    if cache_config is not None:
        from thriftattention.integrations.transformers_cache import ThriftAttentionCache, use_thriftattention_cache

        past = ThriftAttentionCache.from_model(model, config=active_config, max_cache_len=encoded.shape[1] + max_new_tokens)
        past.prefill_real_seq_len = prompt_len
        cache_ctx = use_thriftattention_cache(past)

    generated: list[int] = []
    with torch.inference_mode(), cache_ctx:
        sync_cuda(args.device)
        start = time.perf_counter()
        out = model(input_ids=encoded, use_cache=True, past_key_values=past, cache_position=cache_position)
        sync_cuda(args.device)
        prefill_s = time.perf_counter() - start

        past = out.past_key_values
        if pad:
            _crop_cache(past, prompt_len)
        next_token = out.logits[:, prompt_len - 1, :].argmax(dim=-1)

        sync_cuda(args.device)
        start = time.perf_counter()
        for step in range(max_new_tokens):
            token = int(next_token.item())
            generated.append(token)
            if token == tokenizer.eos_token_id or step == max_new_tokens - 1:
                break
            cache_position = torch.tensor([prompt_len + step], device=args.device, dtype=torch.long)
            out = model(input_ids=next_token[:, None], use_cache=True, past_key_values=past, cache_position=cache_position)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1)
        sync_cuda(args.device)
        decode_s = time.perf_counter() - start

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    decode_steps = max(0, len(generated) - 1)
    return {
        "prediction": text,
        "prompt_tokens": prompt_len,
        "generated_tokens": len(generated),
        "decode_steps": decode_steps,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": prefill_s + decode_s,
        "decode_tok_s": decode_steps / decode_s if decode_s > 0 else None,
        "e2e_tok_s": (prompt_len + len(generated)) / (prefill_s + decode_s),
    }


def _crop_cache(cache: Any, seq_len: int) -> None:
    crop = getattr(cache, "crop", None)
    if callable(crop):
        crop(seq_len)
    layers = getattr(cache, "layers", None)
    if not layers:
        return
    for layer in layers:
        k_fp16 = getattr(layer, "k_fp16", None)
        v_fp16 = getattr(layer, "v_fp16", None)
        if k_fp16 is None or v_fp16 is None:
            continue
        key_end = max(64, ((seq_len + 63) // 64) * 64)
        value_end = ((key_end + 127) // 128) * 128
        key_end = min(key_end, k_fp16.shape[2])
        value_end = min(value_end, v_fp16.shape[2])
        if seq_len < key_end:
            k_fp16[:, :, seq_len:key_end].zero_()
            quantize_k = getattr(layer, "_quantize_k_range", None)
            if callable(quantize_k):
                quantize_k(k_fp16[:, :, seq_len:key_end].contiguous(), seq_len, key_end)
        if seq_len < value_end:
            v_fp16[:, :, seq_len:value_end].zero_()
            quantize_v = getattr(layer, "_quantize_v_range", None)
            if callable(quantize_v):
                quantize_v(seq_len, value_end)


def main() -> None:
    args = parse_args()
    args.lengths = parse_int_list(args.lengths)
    args.tasks = parse_str_list(args.tasks)
    if args.num_examples < 1:
        raise SystemExit("--num-examples must be at least 1")
    try:
        __import__("transformers")
    except Exception:
        raise SystemExit("Missing Transformers. Run `pip install -r examples/long_context_quality/requirements.txt`.")
    ruler_gen, ruler_score = load_ruler_modules(args.ruler_dir)
    args.tasks = normalise_tasks(args.tasks, ruler_gen)

    set_seed(args.seed)
    out_dir = make_output_dir(args.output, prefix="ruler-mini") if args.output is not None else None
    if out_dir is not None:
        write_json(out_dir / "environment.json", collect_environment(args))

    model, tokenizer = load_model(args.model, args.device)
    rows: list[dict[str, Any]] = []

    for run in method_runs(args.methods, args.fractions):
        ok, note, cache_config = configure(model, run["method"], run["fraction"], args.device)
        print(f"\n{run['label']}: {note}")
        if not ok:
            rows.extend({**run, "length": length, "task": task, "status": "skipped", "error": note} for length in args.lengths for task in args.tasks)
            continue
        for length in args.lengths:
            for task in args.tasks:
                samples = ruler_gen.generate_samples(
                    tokenizer=tokenizer,
                    task=task,
                    target_length=length,
                    num_samples=args.num_examples,
                    seed=args.seed,
                    cache_dir=args.cache_dir,
                )
                for index, sample in enumerate(samples):
                    try:
                        result = generate_timed(model, tokenizer, sample, cache_config, args)
                        score = float(ruler_score.score_sample(task, result["prediction"], sample["outputs"]))
                        row = {
                            **run,
                            **result,
                            "length": length,
                            "sample_length": sample.get("length"),
                            "task": task,
                            "example": index,
                            "status": "ok",
                            "outputs": sample["outputs"],
                            "accuracy": score,
                        }
                        rows.append(row)
                        print(
                            f"  len={length:<6} task={task:<17} acc={score:.3f} "
                            f"prefill={row['prefill_s']:.3f}s decode={row['decode_s']:.3f}s total={row['total_s']:.3f}s"
                        )
                    except RuntimeError as exc:
                        rows.append({**run, "length": length, "task": task, "status": "error", "error": str(exc)})
                        print(f"  len={length:<6} task={task:<17} error={exc}")

    summary = average_rows(rows)
    table = markdown_table(
        summary,
        [
            ("tokens", "tokens"),
            ("task", "task"),
            ("method", "method"),
            ("score", "avg_score"),
            ("prefill_s", "prefill_s"),
            ("decode_s", "decode_s"),
            ("total_s", "total_s"),
            ("n", "n"),
            ("status", "status"),
        ],
    )
    print("\nAverage scores")
    print(table)

    if out_dir is not None:
        write_jsonl(out_dir / "metrics.jsonl", rows)
        (out_dir / "summary.md").write_text("# RULER Mini\n\n" + table + "\n", encoding="utf-8")
        print(f"\nWrote {out_dir}")


def average_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["length"], row["task"], row["label"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (length, task, label), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        ok = [item for item in items if item.get("status") == "ok"]
        summary.append(
            {
                "tokens": length,
                "task": task,
                "method": label,
                "score": f"{fmean(item['accuracy'] for item in ok):.3f}" if ok else "-",
                "prefill_s": f"{fmean(item['prefill_s'] for item in ok):.3f}" if ok else "-",
                "decode_s": f"{fmean(item['decode_s'] for item in ok):.3f}" if ok else "-",
                "total_s": f"{fmean(item['total_s'] for item in ok):.3f}" if ok else "-",
                "n": len(ok),
                "status": "ok" if ok else items[0].get("status", "skipped"),
            }
        )
    return summary


if __name__ == "__main__":
    main()
