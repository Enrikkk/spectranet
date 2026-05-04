"""
Shared inference-timing utilities for TIME_ANALYSIS.

Conventions:
  * Single H100, no CUDA Graphs, FP32.
  * model.eval() + torch.no_grad() always.
  * Warmup: 10 untimed iters per (batch_size). Re-warm on every batch-size change.
  * Timing primitive: torch.cuda.Event, with cuda.synchronize before start.
  * 50 timed iters; report median + IQR (p25/p75).
  * Memory: reset_peak_memory_stats() before, max_memory_allocated() after.
  * On OOM: row recorded with status='OOM' + zero numerics, training continues.
"""
import csv
import datetime as _dt
import os
import statistics
from typing import Callable, Iterable, List, Dict, Any

import torch


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def time_forward(
    forward_fn: Callable[[int], None],
    batch_sizes: Iterable[int] = (1, 8, 32),
    warmup: int = 10,
    iters: int = 50,
    extra: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    forward_fn(batch_size) -> None. Closure must build inputs of given batch
    size and run the full unit-of-work forward (single forward for non-AR
    models, full T_out=10 rollout for AR). Closure must call cuda.synchronize
    only if it has reason to; this harness does its own sync.

    Returns list of dicts (one per batch size).
    """
    extra = dict(extra or {})
    rows = []
    device_name = _gpu_name()
    torch_ver = torch.__version__
    for B in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(warmup):
                forward_fn(B)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {
                "batch_size": B,
                "latency_ms_median": 0.0,
                "latency_ms_p25": 0.0,
                "latency_ms_p75": 0.0,
                "throughput_samples_per_sec": 0.0,
                "peak_mem_mb": 0.0,
                "status": "OOM",
                "gpu": device_name,
                "torch_version": torch_ver,
                "timestamp": _now_iso(),
                **extra,
            }
            rows.append(row)
            print(f"  [B={B}] OOM during warmup")
            continue

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        try:
            torch.cuda.synchronize()
            for i in range(iters):
                starts[i].record()
                forward_fn(B)
                ends[i].record()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {
                "batch_size": B,
                "latency_ms_median": 0.0,
                "latency_ms_p25": 0.0,
                "latency_ms_p75": 0.0,
                "throughput_samples_per_sec": 0.0,
                "peak_mem_mb": 0.0,
                "status": "OOM",
                "gpu": device_name,
                "torch_version": torch_ver,
                "timestamp": _now_iso(),
                **extra,
            }
            rows.append(row)
            print(f"  [B={B}] OOM during timed loop")
            continue

        times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        med = statistics.median(times_ms)
        p25 = times_ms[max(0, int(0.25 * iters) - 1)]
        p75 = times_ms[min(iters - 1, int(0.75 * iters))]
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        thpt = (B * 1000.0) / med if med > 0 else 0.0
        row = {
            "batch_size": B,
            "latency_ms_median": med,
            "latency_ms_p25": p25,
            "latency_ms_p75": p75,
            "throughput_samples_per_sec": thpt,
            "peak_mem_mb": peak_mb,
            "status": "ok",
            "gpu": device_name,
            "torch_version": torch_ver,
            "timestamp": _now_iso(),
            **extra,
        }
        rows.append(row)
        print(f"  [B={B}] median={med:.3f}ms  IQR=[{p25:.3f},{p75:.3f}]  "
              f"thpt={thpt:.1f} samp/s  peak={peak_mb:.1f}MB")
    return rows


# Canonical column order for the per-model CSVs.
CSV_COLS = [
    "model", "variant", "params",
    "batch_size",
    "latency_ms_median", "latency_ms_p25", "latency_ms_p75",
    "throughput_samples_per_sec",
    "peak_mem_mb",
    "status",
    "gpu", "torch_version", "timestamp",
]


def write_rows(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    """Append rows to csv_path. Writes header if file is new/empty."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
