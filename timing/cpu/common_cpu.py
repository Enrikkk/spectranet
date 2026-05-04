"""
CPU companion to common.py.

Differences from the GPU harness:
  * timing primitive: time.perf_counter_ns (no cuda.Event)
  * no cuda.synchronize / empty_cache
  * peak_mem_mb is process-RSS delta (psutil), reset before each batch_size
  * device label = CPU model from /proc/cpuinfo
  * tighter default budget (warmup=5, iters=20) — CPU is slow
  * "cpu_threads" recorded from torch.get_num_threads()
"""
import csv
import datetime as _dt
import os
import statistics
import time
from typing import Callable, Iterable, List, Dict, Any

import psutil
import torch


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "cpu"


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def time_forward(
    forward_fn: Callable[[int], None],
    batch_sizes: Iterable[int] = (1, 4),
    warmup: int = 5,
    iters: int = 20,
    extra: Dict[str, Any] = None,
    per_iter_timeout_s: float = None,
) -> List[Dict[str, Any]]:
    """
    forward_fn(batch_size) builds inputs and runs the unit-of-work forward.
    Returns one dict per batch size.

    per_iter_timeout_s: if set and the first warmup iter exceeds this, mark
    the row as 'TOO_SLOW' and skip the timed loop. Lets us cap wall time on
    pathological models (e.g. full attention at large B on CPU).
    """
    extra = dict(extra or {})
    rows = []
    device_name = _cpu_name()
    cpu_threads = torch.get_num_threads()
    torch_ver = torch.__version__
    for B in batch_sizes:
        rss_before = _rss_mb()
        # warmup, with optional timeout on the first iter
        try:
            t0 = time.perf_counter()
            forward_fn(B)
            first_dt = time.perf_counter() - t0
            if per_iter_timeout_s is not None and first_dt > per_iter_timeout_s:
                row = {
                    "batch_size": B,
                    "latency_ms_median": first_dt * 1000.0,
                    "latency_ms_p25": first_dt * 1000.0,
                    "latency_ms_p75": first_dt * 1000.0,
                    "throughput_samples_per_sec": B / first_dt if first_dt > 0 else 0.0,
                    "peak_mem_mb": max(0.0, _rss_mb() - rss_before),
                    "status": "TOO_SLOW",
                    "gpu": device_name,
                    "torch_version": torch_ver,
                    "cpu_threads": cpu_threads,
                    "timestamp": _now_iso(),
                    **extra,
                }
                rows.append(row)
                print(f"  [B={B}] TOO_SLOW first iter {first_dt:.1f}s "
                      f"(>{per_iter_timeout_s}s) — skipping timed loop")
                continue
            for _ in range(max(0, warmup - 1)):
                forward_fn(B)
        except (RuntimeError, MemoryError) as e:
            row = {
                "batch_size": B,
                "latency_ms_median": 0.0,
                "latency_ms_p25": 0.0,
                "latency_ms_p75": 0.0,
                "throughput_samples_per_sec": 0.0,
                "peak_mem_mb": max(0.0, _rss_mb() - rss_before),
                "status": f"ERROR:{type(e).__name__}",
                "gpu": device_name,
                "torch_version": torch_ver,
                "cpu_threads": cpu_threads,
                "timestamp": _now_iso(),
                **extra,
            }
            rows.append(row)
            print(f"  [B={B}] ERROR during warmup: {type(e).__name__}: {e}")
            continue

        rss_peak = _rss_mb()
        times_ms = []
        try:
            for _ in range(iters):
                t0 = time.perf_counter_ns()
                forward_fn(B)
                t1 = time.perf_counter_ns()
                times_ms.append((t1 - t0) / 1e6)
                rss_peak = max(rss_peak, _rss_mb())
        except (RuntimeError, MemoryError) as e:
            row = {
                "batch_size": B,
                "latency_ms_median": 0.0,
                "latency_ms_p25": 0.0,
                "latency_ms_p75": 0.0,
                "throughput_samples_per_sec": 0.0,
                "peak_mem_mb": max(0.0, rss_peak - rss_before),
                "status": f"ERROR:{type(e).__name__}",
                "gpu": device_name,
                "torch_version": torch_ver,
                "cpu_threads": cpu_threads,
                "timestamp": _now_iso(),
                **extra,
            }
            rows.append(row)
            print(f"  [B={B}] ERROR during timed loop: {type(e).__name__}: {e}")
            continue

        times_ms.sort()
        n = len(times_ms)
        med = statistics.median(times_ms)
        p25 = times_ms[max(0, int(0.25 * n) - 1)]
        p75 = times_ms[min(n - 1, int(0.75 * n))]
        peak_mb = max(0.0, rss_peak - rss_before)
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
            "cpu_threads": cpu_threads,
            "timestamp": _now_iso(),
            **extra,
        }
        rows.append(row)
        print(f"  [B={B}] median={med:.2f}ms  IQR=[{p25:.2f},{p75:.2f}]  "
              f"thpt={thpt:.2f} samp/s  rss_delta={peak_mb:.1f}MB")
    return rows


CSV_COLS = [
    "model", "variant", "params",
    "batch_size",
    "latency_ms_median", "latency_ms_p25", "latency_ms_p75",
    "throughput_samples_per_sec",
    "peak_mem_mb",
    "status",
    "gpu", "torch_version", "cpu_threads", "timestamp",
]


def write_rows(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
