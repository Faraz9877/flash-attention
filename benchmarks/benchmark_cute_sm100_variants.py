#!/usr/bin/env python3
import argparse
import importlib
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch


TensorFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]


@dataclass
class VariantResult:
    status: str
    compile_ms: Optional[float] = None
    run_ms: Optional[float] = None
    tflops: Optional[float] = None
    max_abs_err: Optional[float] = None
    nrmse: Optional[float] = None
    message: str = ""


def parse_seq_lens(seq_lens: str) -> list[int]:
    return [int(part.strip()) for part in seq_lens.split(",") if part.strip()]


def parse_dtype(dtype: str) -> torch.dtype:
    value = dtype.lower()
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def compute_tflops(batch: int, heads: int, seq: int, dim: int, causal: bool, ms: float) -> float:
    flops = 4 * batch * heads * seq * seq * dim
    if causal:
        flops //= 2
    return flops / (ms * 1e-3) / 1e12


def compute_error_metrics(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    diff = (out - ref).to(torch.float32)
    ref_f32 = ref.to(torch.float32)
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    ref_rms = torch.sqrt(torch.mean(ref_f32 * ref_f32)).item()
    nrmse = rmse / max(ref_rms, 1e-12)
    max_abs_err = diff.abs().max().item()
    return max_abs_err, nrmse


def unwrap_output(result) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Unsupported output type: {type(result)}")


def short_error(exc: BaseException) -> str:
    msg = str(exc).strip().replace("\n", " ")
    if len(msg) > 160:
        msg = msg[:157] + "..."
    return f"{type(exc).__name__}: {msg}"


def add_repo_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def clear_cute_forward_caches(cute_interface) -> None:
    if hasattr(cute_interface, "_flash_attn_fwd") and hasattr(cute_interface._flash_attn_fwd, "compile_cache"):
        cute_interface._flash_attn_fwd.compile_cache.clear()
    if hasattr(cute_interface, "_flash_attn_fwd_combine") and hasattr(cute_interface._flash_attn_fwd_combine, "compile_cache"):
        cute_interface._flash_attn_fwd_combine.compile_cache.clear()


@contextmanager
def patch_sm100_forward_class(cute_interface, klass):
    original = cute_interface.FlashAttentionForwardSm100
    cute_interface.FlashAttentionForwardSm100 = klass
    try:
        yield
    finally:
        cute_interface.FlashAttentionForwardSm100 = original


@torch.inference_mode()
def benchmark_one(
    fn: TensorFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float, float]:
    compile_start = time.perf_counter()
    out = unwrap_output(fn(q, k, v, causal))
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    for _ in range(max(warmup - 1, 0)):
        out = unwrap_output(fn(q, k, v, causal))
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = unwrap_output(fn(q, k, v, causal))
    end.record()
    torch.cuda.synchronize()
    run_ms = start.elapsed_time(end) / iters
    return out, compile_ms, run_ms


def make_variant_fn(cute_interface) -> TensorFn:
    def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
        out = cute_interface.flash_attn_func(q, k, v, causal=causal)
        return unwrap_output(out)

    return run


def run_variant(
    name: str,
    cute_interface,
    klass,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    warmup: int,
    iters: int,
    ref: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], VariantResult]:
    try:
        clear_cute_forward_caches(cute_interface)
        with patch_sm100_forward_class(cute_interface, klass):
            out, compile_ms, run_ms = benchmark_one(make_variant_fn(cute_interface), q, k, v, causal, warmup, iters)
        result = VariantResult(
            status="OK",
            compile_ms=compile_ms,
            run_ms=run_ms,
            tflops=compute_tflops(q.shape[0], q.shape[2], q.shape[1], q.shape[3], causal, run_ms),
        )
        if ref is not None:
            result.max_abs_err, result.nrmse = compute_error_metrics(out, ref)
        return out, result
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return None, VariantResult(status="OOM", message=short_error(exc))
    except Exception as exc:
        return None, VariantResult(status="SKIP", message=short_error(exc))


def should_run_variant(name: str, filter_expr: str) -> bool:
    if not filter_expr or filter_expr.lower() == "all":
        return True
    filters = [token.strip().lower() for token in filter_expr.split(",") if token.strip()]
    return any(token in name.lower() for token in filters)


def print_result_row(seq: int, name: str, result: VariantResult, baseline_ms: Optional[float]) -> None:
    speedup = ""
    if baseline_ms is not None and result.run_ms and result.status == "OK":
        speedup = f"{(baseline_ms / result.run_ms):7.2f}x"
    elif baseline_ms is not None:
        speedup = "   n/a "

    compile_col = f"{result.compile_ms:9.3f}" if result.compile_ms is not None else "    -    "
    run_col = f"{result.run_ms:9.3f}" if result.run_ms is not None else "    -    "
    tflops_col = f"{result.tflops:8.2f}" if result.tflops is not None else "   -    "
    err_col = f"{result.max_abs_err:10.3e}" if result.max_abs_err is not None else "    -     "
    nrmse_col = f"{result.nrmse:10.3e}" if result.nrmse is not None else "    -     "
    print(f"{seq:6d}  {name:28s}  {result.status:4s}  {compile_col}  {run_col}  {tflops_col}  {speedup:>7s}  {err_col}  {nrmse_col}  {result.message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark default vs custom CuTe SM100 forward kernel")
    parser.add_argument("--seq-lens", default="1024,2048,4096")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--variants", default="default,custom")
    parser.add_argument("--custom-module", default="flash_attn.cute.flash_fwd_sm100_sageexp")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        raise RuntimeError(f"This benchmark is intended for SM100; found sm{major}{minor}")

    add_repo_to_path()
    cute_interface = importlib.import_module("flash_attn.cute.interface")
    custom_mod = importlib.import_module(args.custom_module)

    default_class = cute_interface.FlashAttentionForwardSm100
    custom_class = custom_mod.FlashAttentionForwardSm100

    variants: list[tuple[str, object]] = [
        ("flash.cute.sm100.default", default_class),
        ("flash.cute.sm100.custom", custom_class),
    ]
    variants = [(name, klass) for name, klass in variants if should_run_variant(name, args.variants)]
    if not variants:
        raise ValueError(f"No variants selected by --variants={args.variants}")

    dtype = parse_dtype(args.dtype)
    print("seq     method                         st    compile_ms     run_ms    TFLOPs   speedup   max_abs_err   nrmse      message")
    print("-" * 134)

    for seq in parse_seq_lens(args.seq_lens):
        q = torch.randn(args.batch, seq, args.heads, args.dim, device=device, dtype=dtype)
        k = torch.randn(args.batch, seq, args.heads, args.dim, device=device, dtype=dtype)
        v = torch.randn(args.batch, seq, args.heads, args.dim, device=device, dtype=dtype)

        default_out: Optional[torch.Tensor] = None
        default_result: Optional[VariantResult] = None
        for name, klass in variants:
            ref = default_out if (name.endswith(".custom") and default_out is not None) else None
            out, result = run_variant(name, cute_interface, klass, q, k, v, args.causal, args.warmup, args.iters, ref)
            if name.endswith(".default"):
                default_out = out.detach().clone() if out is not None else None
                default_result = result
            baseline_ms = default_result.run_ms if (default_result and default_result.status == "OK") else None
            print_result_row(seq, name, result, baseline_ms)


if __name__ == "__main__":
    main()
