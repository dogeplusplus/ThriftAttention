#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

import torch

import thriftattention as ta
from thriftattention._extension import get_extension
from thriftattention.quantization import nvfp4_quantize, nvfp4_quantize_transposed
from thriftattention.selection import resolve_top_k


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_fractions(raw: str) -> list[float]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item[:-1]) / 100.0 if item.endswith("%") else float(item))
    return out


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def round_up(a: int, b: int) -> int:
    return cdiv(a, b) * b


def time_cuda(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def group_q(q: torch.Tensor, kv_heads: int) -> torch.Tensor:
    bsz, q_heads, q_len, dim = q.shape
    if q_len != 1:
        raise ValueError("decode benchmark expects q_len=1")
    return q.reshape(bsz, kv_heads, q_heads // kv_heads, dim).contiguous()


def cached_k_mean(k_cache: torch.Tensor, seq_len: int) -> torch.Tensor:
    bsz, kv_heads, _, dim = k_cache.shape
    blocks = seq_len // 64
    return (
        k_cache[:, :, : blocks * 64]
        .reshape(bsz, kv_heads, blocks, 64, dim)
        .float()
        .mean(dim=3)
        .to(torch.float16)
        .contiguous()
    )


def select_topk(q_grouped: torch.Tensor, k_mean: torch.Tensor, fraction: float) -> torch.Tensor:
    bsz, kv_heads, groups, dim = q_grouped.shape
    blocks = k_mean.shape[2]
    top_k = resolve_top_k(blocks, causal=False, fraction=fraction)
    if top_k == 0:
        return torch.empty(bsz * kv_heads, 0, device=q_grouped.device, dtype=torch.int32)

    scores = (q_grouped.float().unsqueeze(3) * k_mean.float().unsqueeze(2)).sum(dim=-1)
    scores = scores.amax(dim=2).reshape(bsz * kv_heads, blocks)
    return scores.topk(top_k, dim=-1).indices.to(torch.int32).contiguous()


def update_one_token(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_packed: torch.Tensor,
    k_scale: torch.Tensor,
    v_packed_t: torch.Tensor,
    v_scale_t: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    pos: int,
) -> None:
    k_cache[:, :, pos : pos + 1] = k_new
    v_cache[:, :, pos : pos + 1] = v_new

    k_p, k_s = nvfp4_quantize(k_new.contiguous())
    k_packed[:, :, pos : pos + 1] = k_p
    k_scale[:, :, pos : pos + 1] = k_s

    # V is transposed and scaled over 16-token sequence groups, so a one-token
    # append re-quantizes the touched 16-token V group.
    begin = (pos // 16) * 16
    end = begin + 16
    v_p, v_s = nvfp4_quantize_transposed(v_cache[:, :, begin:end].contiguous())
    v_packed_t[:, :, :, begin // 2 : end // 2] = v_p[:, :, :, :8]
    v_scale_t[:, :, :, begin // 16 : end // 16] = v_s[:, :, :, :1]


def refresh_one_k_mean(k_cache: torch.Tensor, k_mean: torch.Tensor, block: int) -> None:
    start = block * 64
    k_mean[:, :, block] = (
        k_cache[:, :, start : start + 64].float().mean(dim=2).to(torch.float16)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark-only fair decode profile with cached packed K/V and cached K means."
    )
    parser.add_argument("--seq-lens", default="32768,65536,131072")
    parser.add_argument("--fractions", default="0,1%,5%,10%,25%")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-stateless", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.heads % args.kv_heads != 0:
        raise SystemExit("--heads must be divisible by --kv-heads")

    seq_lens = parse_ints(args.seq_lens)
    fractions = parse_fractions(args.fractions)
    ext = get_extension()
    rows = []

    for seq_len in seq_lens:
        if seq_len % 64 != 0:
            raise ValueError(f"seq_len must be divisible by 64, got {seq_len}")

        gen = torch.Generator(device="cuda")
        gen.manual_seed(args.seed + seq_len)
        q = torch.randn(
            args.batch_size, args.heads, 1, args.head_dim,
            device="cuda", dtype=torch.float16, generator=gen,
        )
        k = torch.randn(
            args.batch_size, args.kv_heads, seq_len, args.head_dim,
            device="cuda", dtype=torch.float16, generator=gen,
        )
        v = torch.randn_like(k)

        cap = round_up(seq_len + 128, 128)
        k_cache = torch.zeros(args.batch_size, args.kv_heads, cap, args.head_dim, device="cuda", dtype=torch.float16)
        v_cache = torch.zeros_like(k_cache)
        k_cache[:, :, :seq_len] = k
        v_cache[:, :, :seq_len] = v

        k_packed = torch.zeros(args.batch_size, args.kv_heads, cap, args.head_dim // 2, device="cuda", dtype=torch.uint8)
        k_scale = torch.zeros(args.batch_size, args.kv_heads, cap, args.head_dim // 16, device="cuda", dtype=torch.float8_e4m3fn)
        v_packed_t = torch.zeros(args.batch_size, args.kv_heads, args.head_dim, cap // 2, device="cuda", dtype=torch.uint8)
        v_scale_t = torch.zeros(args.batch_size, args.kv_heads, args.head_dim, cap // 16, device="cuda", dtype=torch.float8_e4m3fn)

        k_p, k_s = nvfp4_quantize(k)
        v_p, v_s = nvfp4_quantize_transposed(v)
        k_packed[:, :, :seq_len] = k_p
        k_scale[:, :, :seq_len] = k_s
        v_packed_t[:, :, :, : v_p.shape[-1]] = v_p
        v_scale_t[:, :, :, : v_s.shape[-1]] = v_s

        k_mean = torch.zeros(
            args.batch_size,
            args.kv_heads,
            cap // 64,
            args.head_dim,
            device="cuda",
            dtype=torch.float16,
        )
        k_mean[:, :, : seq_len // 64] = cached_k_mean(k_cache, seq_len)
        q_grouped = group_q(q, args.kv_heads)
        q_packed, q_scale = nvfp4_quantize(q_grouped)

        k_new = torch.randn(args.batch_size, args.kv_heads, 1, args.head_dim, device="cuda", dtype=torch.float16)
        v_new = torch.randn_like(k_new)

        q_quant_ms = time_cuda(lambda: nvfp4_quantize(q_grouped), args.warmup, args.repeat)
        update_ms = time_cuda(
            lambda: update_one_token(k_cache, v_cache, k_packed, k_scale, v_packed_t, v_scale_t, k_new, v_new, seq_len),
            args.warmup,
            args.repeat,
        )
        update_boundary_ms = time_cuda(
            lambda: (
                update_one_token(k_cache, v_cache, k_packed, k_scale, v_packed_t, v_scale_t, k_new, v_new, seq_len + 63),
                refresh_one_k_mean(k_cache, k_mean, seq_len // 64),
            ),
            args.warmup,
            args.repeat,
        )

        print(
            f"seq={seq_len} blocks={seq_len // 64} "
            f"q_quant={q_quant_ms:.4f}ms update={update_ms:.4f}ms "
            f"update+mean={update_boundary_ms:.4f}ms"
        )

        for fraction in fractions:
            k_mean_active = k_mean[:, :, : seq_len // 64]
            selected = select_topk(q_grouped, k_mean_active, fraction)

            def kernel_call() -> torch.Tensor:
                return ext.thrift_attention_single_query_nvfp4_packed(
                    q_grouped,
                    k_cache[:, :, :seq_len],
                    v_cache[:, :, :seq_len],
                    selected,
                    q_packed,
                    k_packed[:, :, :seq_len],
                    v_packed_t[:, :, :, : round_up(seq_len, 128) // 2],
                    q_scale,
                    k_scale[:, :, :seq_len],
                    v_scale_t[:, :, :, : round_up(seq_len, 128) // 16],
                )

            def cached_decode() -> torch.Tensor:
                q_p, q_s = nvfp4_quantize(q_grouped)
                topk = select_topk(q_grouped, k_mean_active, fraction)
                return ext.thrift_attention_single_query_nvfp4_packed(
                    q_grouped,
                    k_cache[:, :, :seq_len],
                    v_cache[:, :, :seq_len],
                    topk,
                    q_p,
                    k_packed[:, :, :seq_len],
                    v_packed_t[:, :, :, : round_up(seq_len, 128) // 2],
                    q_s,
                    k_scale[:, :, :seq_len],
                    v_scale_t[:, :, :, : round_up(seq_len, 128) // 16],
                )

            select_ms = time_cuda(lambda: select_topk(q_grouped, k_mean_active, fraction), args.warmup, args.repeat)
            kernel_ms = time_cuda(kernel_call, args.warmup, args.repeat)
            cached_ms = time_cuda(cached_decode, args.warmup, args.repeat)

            stateless_ms = None
            speedup = None
            if not args.no_stateless:
                stateless_ms = time_cuda(
                    lambda: ta.attention(q, k, v, fraction=fraction, _implementation="single_query"),
                    max(1, args.warmup // 2),
                    max(3, args.repeat // 2),
                )
                speedup = stateless_ms / cached_ms

            row = {
                "seq_len": seq_len,
                "kv_blocks": seq_len // 64,
                "fraction": fraction,
                "top_k": selected.shape[1],
                "q_quant_ms": q_quant_ms,
                "topk_select_ms": select_ms,
                "decode_kernel_ms": kernel_ms,
                "cached_decode_ms": cached_ms,
                "cache_update_ms": update_ms,
                "cache_update_boundary_ms": update_boundary_ms,
                "stateless_attention_ms": stateless_ms,
                "cached_speedup_vs_stateless": speedup,
            }
            rows.append(row)
            speedup_text = "-" if speedup is None else f"{speedup:.2f}x"
            print(
                f"  topk@{fraction * 100:g}% k={selected.shape[1]:4d} "
                f"select={select_ms:.4f}ms kernel={kernel_ms:.4f}ms "
                f"cached={cached_ms:.4f}ms speedup_vs_stateless={speedup_text}"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
