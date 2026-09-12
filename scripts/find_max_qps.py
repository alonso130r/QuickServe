#!/usr/bin/env python3
"""Quickly estimate the highest sustainable QuickServe benchmark QPS."""

import argparse
import json
import math
import numbers
import pathlib
import subprocess
import sys
import tempfile
from typing import Callable, NamedTuple


class ProbeResult(NamedTuple):
    qps: float
    achieved_qps: float
    ttft_p99_ns: int
    tpot_p99_ns: int
    peak_queued: int
    final_active: int
    final_queued: int
    failed: int
    rejected: int
    sustainable: bool


def is_sustainable(target_qps, achieved_qps, ttft_p99_ns, tpot_p99_ns,
                   peak_queued, final_active, final_queued, failed, rejected,
                   requests, throughput_ratio, queue_fraction,
                   max_p99_ttft_ns, max_p99_tpot_ns):
    queue_limit = max(8, math.floor(requests * queue_fraction))
    return (achieved_qps >= target_qps * throughput_ratio
            and ttft_p99_ns <= max_p99_ttft_ns
            and tpot_p99_ns <= max_p99_tpot_ns
            and peak_queued <= queue_limit
            and final_active == 0
            and final_queued == 0
            and failed == 0
            and rejected == 0)


def probe_result_from_summary(target_qps, summary, args):
    def nonnegative_number(value):
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(value) or value < 0):
            raise ValueError
        return value

    def nonnegative_count(value):
        number = nonnegative_number(value)
        if not float(number).is_integer():
            raise ValueError
        return int(number)

    try:
        achieved_qps = nonnegative_number(summary["achieved_request_qps"])
        ttft_p99_ns = nonnegative_count(summary["ttft"]["p99_ns"])
        tpot_p99_ns = nonnegative_count(summary["tpot"]["p99_ns"])
        peak_queued = nonnegative_count(summary["peak_queued_requests"])
        final_active = nonnegative_count(summary["final_active_requests"])
        final_queued = nonnegative_count(summary["final_queued_requests"])
        failed = nonnegative_count(summary["counts"]["failed"])
        rejected = nonnegative_count(summary["counts"]["rejected"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "benchmark summary is missing or has invalid sustainability metrics"
        ) from error
    sustainable = is_sustainable(
        target_qps, achieved_qps, ttft_p99_ns, tpot_p99_ns, peak_queued,
        final_active, final_queued, failed, rejected, args.requests,
        args.throughput_ratio, args.max_queue_fraction,
        args.max_p99_ttft_ms * 1_000_000,
        args.max_p99_tpot_ms * 1_000_000)
    return ProbeResult(
        target_qps, achieved_qps, ttft_p99_ns, tpot_p99_ns, peak_queued,
        final_active, final_queued, failed, rejected, sustainable)


def find_max_qps(probe: Callable[[float], ProbeResult], start_qps: float,
                 max_qps: float, binary_steps: int) -> ProbeResult:
    first = probe(start_qps)
    if not first.sustainable:
        raise RuntimeError("starting QPS is not sustainable; lower --start-qps")

    lower = first
    while lower.qps < max_qps:
        candidate = min(max_qps, lower.qps * 2)
        result = probe(candidate)
        if result.sustainable:
            lower = result
            if candidate == max_qps:
                return lower
            continue

        upper = result
        for _ in range(binary_steps):
            midpoint = (lower.qps + upper.qps) / 2
            result = probe(midpoint)
            if result.sustainable:
                lower = result
            else:
                upper = result
        return lower
    return lower


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate maximum sustainable QPS with an exponential sweep "
                    "followed by binary search.")
    parser.add_argument("--benchmark", default="build-benchmark/quickserve_benchmark")
    parser.add_argument("--trace", default="data/AzureLLMInferenceTrace_code_1week.qst")
    parser.add_argument("--model", required=True)
    parser.add_argument("--start-qps", type=float, default=2.0)
    parser.add_argument("--max-qps", type=float, default=128.0)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--binary-steps", type=int, default=4)
    parser.add_argument("--throughput-ratio", type=float, default=0.98)
    parser.add_argument("--max-queue-fraction", type=float, default=0.25)
    parser.add_argument("--max-p99-ttft-ms", type=float, default=2000)
    parser.add_argument("--max-p99-tpot-ms", type=float, default=200)
    parser.add_argument("--output-mode", choices=("natural", "trace-exact"),
                        default="trace-exact")
    parser.add_argument("--context-size", type=int, default=16384)
    parser.add_argument("--batch-capacity", type=int, default=512)
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--work-dir", type=pathlib.Path,
                        help="Keep each probe's full benchmark results here")
    return parser.parse_args()


def validate_args(args):
    if not (0 < args.start_qps <= args.max_qps):
        raise ValueError("require 0 < --start-qps <= --max-qps")
    if args.requests <= 0 or args.binary_steps < 0:
        raise ValueError("--requests must be positive and --binary-steps nonnegative")
    if not (0 < args.throughput_ratio <= 1):
        raise ValueError("--throughput-ratio must be in (0, 1]")
    if not (0 <= args.max_queue_fraction <= 1):
        raise ValueError("--max-queue-fraction must be in [0, 1]")
    if not math.isfinite(args.max_p99_ttft_ms) or args.max_p99_ttft_ms <= 0:
        raise ValueError("--max-p99-ttft-ms (TTFT) must be finite and positive")
    if not math.isfinite(args.max_p99_tpot_ms) or args.max_p99_tpot_ms <= 0:
        raise ValueError("--max-p99-tpot-ms (TPOT) must be finite and positive")


def main():
    args = parse_args()
    try:
        validate_args(args)
        benchmark = pathlib.Path(args.benchmark).resolve(strict=True)
        trace = pathlib.Path(args.trace).resolve(strict=True)
        model = pathlib.Path(args.model).resolve(strict=True)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    temporary = None
    if args.work_dir:
        root = args.work_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="quickserve-qps-search-")
        root = pathlib.Path(temporary.name)

    probe_number = 0
    results = []

    def probe(qps):
        nonlocal probe_number
        probe_number += 1
        output = root / f"probe-{probe_number:02d}-qps-{qps:.4f}"
        command = [
            str(benchmark), "--trace", str(trace), "--model", str(model),
            "--target-qps", f"{qps:.12g}", "--max-requests", str(args.requests),
            "--output-mode", args.output_mode, "--output-dir", str(output),
            "--context-size", str(args.context_size),
            "--batch-capacity", str(args.batch_capacity),
            "--max-sequences", str(args.max_sequences),
            "--token-budget", str(args.token_budget),
        ]
        print(f"[{probe_number}] testing {qps:.3f} QPS ... ", end="", flush=True)
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"benchmark probe failed at {qps:g} QPS: {detail}")
        with (output / "summary.json").open(encoding="utf-8") as stream:
            summary = json.load(stream)
        result = probe_result_from_summary(qps, summary, args)
        results.append(result)
        print(f"achieved {result.achieved_qps:.3f}, peak queue "
              f"{result.peak_queued}: {'PASS' if result.sustainable else 'FAIL'}")
        return result

    try:
        best = find_max_qps(probe, args.start_qps, args.max_qps, args.binary_steps)
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            temporary.cleanup()

    print(f"\nRecommended maximum sustainable QPS: {best.qps:.3f}")
    print(f"Measured achieved QPS: {best.achieved_qps:.3f}; "
          f"peak queued requests: {best.peak_queued}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
