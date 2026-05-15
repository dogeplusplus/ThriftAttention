#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

import torch

import thriftattention as ta


TimedFn = Callable[[], Any]


@dataclass(frozen=True)
class Method:
    name: str
    mode: str
    fraction: float | None = None


def parse_fraction(raw: str) -> float:
    raw = raw.strip()
    value = float(raw[:-1]) / 100.0 if raw.endswith("%") else float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("--fraction must be in [0, 1], or a percentage")
    return value


def thrift_name(fraction: float) -> str:
    pct = fraction * 100.0
    return f"thrift_{int(pct)}pct" if pct.is_integer() else f"thrift_{pct:g}pct".replace(".", "p")


def parse_methods(raw: str, fraction: float) -> list[Method]:
    aliases = {
        "flash": "fp16",
        "fp16_flash": "fp16",
        "fp16-flash": "fp16",
        "ta": "thrift",
        "thriftattention": "thrift",
        "fp5": "fp4",
    }
    methods: list[Method] = []
    for item in raw.split(","):
        key = aliases.get(item.strip().lower(), item.strip().lower())
        if not key:
            continue
        if key == "fp16":
            methods.append(Method("fp16_flash", "fp16"))
        elif key == "fp4":
            methods.append(Method("fp4", "fp4"))
        elif key == "thrift":
            methods.append(Method(thrift_name(fraction), "thrift", fraction))
        else:
            raise argparse.ArgumentTypeError(f"unknown method {item!r}")
    if not methods:
        raise argparse.ArgumentTypeError("at least one method is required")
    return methods


def timed(fn: TimedFn, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def load_model(args: argparse.Namespace) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.to(args.device)
    model.eval()
    return model


def configure(model: Any, method: Method, args: argparse.Namespace) -> None:
    try:
        ta.unpatch_model(model, backend="hf")
    except Exception:
        pass

    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter) and args.attn_implementation:
        setter(args.attn_implementation)

    if method.mode != "fp16":
        ta.patch_model(
            model,
            backend="hf",
            mode=method.mode,
            causal=True,
            fp16_fraction=0.0 if method.fraction is None else method.fraction,
            patch_generation=True,
        )


def input_ids_for(model: Any, args: argparse.Namespace) -> torch.Tensor:
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 32000))
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    return torch.randint(
        1,
        max(2, vocab_size - 1),
        (args.batch_size, args.context_len),
        device=args.device,
        dtype=torch.long,
        generator=generator,
    )


def generate_kwargs(model: Any, input_ids: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    gen_config = getattr(model, "generation_config", None)
    pad_token_id = getattr(gen_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(gen_config, "eos_token_id", None)
    return {
        "input_ids": input_ids,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": pad_token_id,
    }


def phase_fns(model: Any, input_ids: torch.Tensor, args: argparse.Namespace) -> dict[str, TimedFn]:
    gen_kwargs = generate_kwargs(model, input_ids, args)

    def forward() -> Any:
        with torch.inference_mode():
            return model(input_ids=input_ids, use_cache=False)

    def generate() -> Any:
        with torch.inference_mode():
            return model.generate(**gen_kwargs)

    def e2e() -> Any:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
            return model.generate(**gen_kwargs)

    return {"forward": forward, "generate": generate, "e2e": e2e}


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["method", "phase", "median_ms", "mean_ms", "min_ms", "max_ms"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))
    print()
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple HF benchmark for forward, generate, and e2e timings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--context-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--methods", default="fp16,fp4,thrift")
    parser.add_argument("--fraction", type=parse_fraction, default=parse_fraction("5%"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/hf_simple/timings.csv"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.context_len <= 0 or args.context_len % 64 != 0:
        raise SystemExit("--context-len must be positive and divisible by 64")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    args.methods = parse_methods(args.methods, args.fraction)
    return args


def main() -> None:
    args = parse_args()
    model = load_model(args)
    input_ids = input_ids_for(model, args)
    rows: list[dict[str, Any]] = []

    for method in args.methods:
        configure(model, method, args)
        fns = phase_fns(model, input_ids, args)
        print(f"\n{method.name}")
        for phase in ("forward", "generate", "e2e"):
            stats = timed(fns[phase], args.warmup, args.repeat)
            row = {
                "method": method.name,
                "phase": phase,
                "median_ms": f"{stats['median_ms']:.3f}",
                "mean_ms": f"{stats['mean_ms']:.3f}",
                "min_ms": f"{stats['min_ms']:.3f}",
                "max_ms": f"{stats['max_ms']:.3f}",
            }
            rows.append(row)
            print(f"  {phase}: {row['median_ms']} ms")

    print_table(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
