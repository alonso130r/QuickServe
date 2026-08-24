#!/usr/bin/env python3
"""Estimate device capacity with one saturated QuickServe benchmark run."""

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(
        description="Load the model once, saturate it, and estimate sustainable QPS."
    )
    parser.add_argument("--benchmark", default="build-benchmark/quickserve_benchmark")
    parser.add_argument("--trace", default="data/AzureLLMInferenceTrace_code_1week.qst")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--probe-qps", type=float, default=1000.0)
    parser.add_argument("--safety-margin", type=float, default=0.90)
    parser.add_argument(
        "--output-mode", choices=("natural", "trace-exact"), default="trace-exact"
    )
    parser.add_argument("--context-size", type=int, default=16384)
    parser.add_argument("--batch-capacity", type=int, default=512)
    parser.add_argument("--max-sequences", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Keep requests.csv and summary.json in this new directory",
    )
    args = parser.parse_args()

    if args.requests <= 0 or args.probe_qps <= 0:
        parser.error("--requests and --probe-qps must be positive")
    if not 0 < args.safety_margin <= 1:
        parser.error("--safety-margin must be in (0, 1]")

    try:
        benchmark = pathlib.Path(args.benchmark).resolve(strict=True)
        trace = pathlib.Path(args.trace).resolve(strict=True)
        model = pathlib.Path(args.model).resolve(strict=True)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    temporary = None
    if args.output_dir:
        output = args.output_dir.resolve()
    else:
        temporary = tempfile.TemporaryDirectory(prefix="quickserve-capacity-")
        output = pathlib.Path(temporary.name) / "result"
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(benchmark),
        "--trace",
        str(trace),
        "--model",
        str(model),
        "--target-qps",
        str(args.probe_qps),
        "--max-requests",
        str(args.requests),
        "--output-mode",
        args.output_mode,
        "--output-dir",
        str(output),
        "--context-size",
        str(args.context_size),
        "--batch-capacity",
        str(args.batch_capacity),
        "--max-sequences",
        str(args.max_sequences),
        "--token-budget",
        str(args.token_budget),
    ]

    print(f"Loading model once and testing {args.requests} requests...", flush=True)
    started = time.monotonic()
    completed_rows = 0
    csv_offset = 0
    saw_header = False
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        temporary_output = output.parent / f".{output.name}.tmp-{process.pid}"

        while process.poll() is None:
            requests_csv = temporary_output / "requests.csv"
            if requests_csv.exists():
                with requests_csv.open("rb") as stream:
                    stream.seek(csv_offset)
                    chunk = stream.read()
                    csv_offset += len(chunk)
                lines = chunk.count(b"\n")
                if not saw_header and lines:
                    lines -= 1
                    saw_header = True
                completed_rows += lines

            elapsed = max(time.monotonic() - started, 0.001)
            rate = completed_rows / elapsed
            remaining = max(args.requests - completed_rows, 0)
            eta = remaining / rate if rate else None
            eta_text = f"{eta / 60:.1f}m" if eta is not None else "--"
            percent = min(100.0, completed_rows * 100.0 / args.requests)
            print(
                f"\rProgress: {completed_rows}/{args.requests} ({percent:5.1f}%) | "
                f"elapsed {elapsed / 60:.1f}m | ETA {eta_text}",
                end="",
                flush=True,
            )
            time.sleep(2)

        returncode = process.wait()
        print()
        if returncode != 0:
            log.seek(0)
            detail = log.read().strip()
        else:
            detail = ""

    if returncode != 0:
        if detail:
            print(detail, file=sys.stderr)
        if temporary:
            temporary.cleanup()
        return returncode

    try:
        with (output / "summary.json").open(encoding="utf-8") as stream:
            summary = json.load(stream)
        measured = float(summary["achieved_request_qps"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: cannot read benchmark result: {error}", file=sys.stderr)
        if temporary:
            temporary.cleanup()
        return 1

    recommended = measured * args.safety_margin
    print(f"Maximum measured QPS: {measured:.3f}")
    print(
        f"Recommended sustained target ({args.safety_margin:.0%}): "
        f"{recommended:.3f} QPS"
    )
    if temporary:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
