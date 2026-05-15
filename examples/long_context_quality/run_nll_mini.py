#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from contextlib import nullcontext
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
    thrift_acceleration_status,
    timed_call,
    write_json,
    write_jsonl,
)


DEFAULT_MODEL = "Qwen/Qwen3-8B"
METHODS = {"fp16": "fp16", "flash": "fp16", "fp4": "fp4", "thrift": "thrift"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini forward/NLL benchmark for patched HF models.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lengths", default="4096,8192")
    parser.add_argument("--methods", default="fp16,fp4,thrift")
    parser.add_argument("--fractions", default="0.05")
    parser.add_argument("--num-docs", type=int, default=1)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--ce-chunk", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("results/long_context_quality"))
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
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return model, tokenizer


def configure(model: Any, method: str, fraction: float | None, device: str) -> tuple[bool, str]:
    import thriftattention as ta

    try:
        ta.unpatch_model(model, backend="hf")
    except Exception:
        pass
    if method == "fp16":
        return True, "standard Transformers attention"
    ready, note = thrift_acceleration_status(device)
    if not ready:
        return False, note
    try:
        ta.patch_model(
            model,
            backend="hf",
            mode="fp4" if method == "fp4" else "thrift",
            fp16_fraction=0.0 if fraction is None else fraction,
            patch_generation=False,
        )
    except Exception as exc:
        return False, str(exc)
    return True, f"patched {method}" if method == "fp4" else f"patched thrift fraction={fraction:g}"


def make_docs(tokenizer: Any, args: argparse.Namespace, max_len: int) -> list[list[int]]:
    text = (
        args.text_file.read_text(encoding="utf-8")
        if args.text_file
        else "ThriftAttention mini NLL text. This local synthetic corpus avoids dataset downloads. "
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] or [tokenizer.eos_token_id or 0]
    ids = (ids * math.ceil(max_len / len(ids)))[:max_len]
    return [ids[:] for _ in range(args.num_docs)]


def body(model: Any) -> Any:
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix)
    for name in ("model", "transformer", "gpt_neox", "backbone"):
        if hasattr(model, name):
            return getattr(model, name)
    raise RuntimeError("could not find transformer body")


def mean_nll(model: Any, ids: list[int], length: int, args: argparse.Namespace) -> float:
    import torch
    import torch.nn.functional as F

    input_ids = torch.tensor([ids[:length]], dtype=torch.long, device=args.device)
    targets = torch.tensor(ids[1:length], dtype=torch.long, device=args.device)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    autocast = torch.autocast("cuda", dtype=dtype) if args.device.startswith("cuda") else nullcontext()

    with torch.inference_mode(), autocast:
        hidden = body(model)(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state[:, :-1, :]
        total = 0.0
        for start in range(0, targets.numel(), args.ce_chunk):
            end = min(start + args.ce_chunk, targets.numel())
            logits = model.get_output_embeddings()(hidden[:, start:end, :]).squeeze(0).float()
            total += float(F.cross_entropy(logits, targets[start:end], reduction="sum"))
    return total / targets.numel()


def main() -> None:
    args = parse_args()
    args.lengths = parse_int_list(args.lengths)
    if any(length < 2 for length in args.lengths):
        raise SystemExit("--lengths must be at least 2")
    if args.num_docs < 1:
        raise SystemExit("--num-docs must be at least 1")
    try:
        __import__("transformers")
    except Exception:
        raise SystemExit("Missing Transformers. Run `pip install -r examples/long_context_quality/requirements.txt`.")

    set_seed(args.seed)
    out_dir = make_output_dir(args.output, prefix="nll-mini")
    write_json(out_dir / "environment.json", collect_environment(args))

    model, tokenizer = load_model(args.model, args.device)
    docs = make_docs(tokenizer, args, max(args.lengths))

    rows: list[dict[str, Any]] = []
    for run in method_runs(args.methods, args.fractions):
        ok, note = configure(model, run["method"], run["fraction"], args.device)
        print(f"\n{run['label']}: {note}")
        if not ok:
            rows.extend({**run, "length": length, "status": "skipped", "error": note} for length in args.lengths)
            continue
        for length in args.lengths:
            values, seconds = [], []
            for doc in docs:
                value, elapsed = timed_call(lambda: mean_nll(model, doc, length, args), device=args.device)
                values.append(value)
                seconds.append(elapsed)
            row = {
                **run,
                "length": length,
                "status": "ok",
                "mean_nll": fmean(values),
                "ppl": math.exp(fmean(values)) if fmean(values) < 50 else float("inf"),
                "forward_s": fmean(seconds),
                "forward_tok_s": length / fmean(seconds),
            }
            rows.append(row)
            print(f"  length={length:<6} nll={row['mean_nll']:.4f} forward_s={row['forward_s']:.3f}")

    baselines = {row["length"]: row["mean_nll"] for row in rows if row.get("label") == "fp16" and row.get("status") == "ok"}
    for row in rows:
        if row.get("status") == "ok" and row["length"] in baselines:
            row["delta_vs_fp16"] = row["mean_nll"] - baselines[row["length"]]

    write_jsonl(out_dir / "metrics.jsonl", rows)
    table = markdown_table(
        rows,
        [
            ("length", "tokens"),
            ("label", "method"),
            ("mean_nll", "mean_nll"),
            ("delta_vs_fp16", "delta"),
            ("forward_s", "forward_s"),
            ("forward_tok_s", "tok/s"),
            ("status", "status"),
        ],
    )
    (out_dir / "summary.md").write_text("# NLL Mini\n\n" + table + "\n", encoding="utf-8")
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
