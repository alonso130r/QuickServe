#!/usr/bin/env python3
"""Run and aggregate controlled, matched-load QuickServe policy comparisons."""

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import NamedTuple


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY_FILES = {
    "fifo": "fifo",
    "continuous_fifo": "continuous_batched_fifo",
    "aimd": "heuristic_aimd",
    "proxqp": "proxqp_scheduler",
}
# A trace-window identity becomes one output-directory segment. Restrict it to
# an ASCII alphanumeric prefix followed by ASCII alphanumerics, dot, dash, or
# underscore so it cannot traverse directories or introduce control characters.
TRACE_WINDOW_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
REPORT_COLUMNS = [
    "policy", "offered_qps", "trace_window", "repetition",
    "achieved_qps", "ttft_p50_ns", "ttft_p95_ns", "ttft_p99_ns",
    "tpot_p99_ns", "peak_queue_depth", "jain_fairness",
    "scheduler_decision_count", "scheduler_mean_ns", "scheduler_p95_ns",
    "scheduler_p99_ns", "peak_rss_bytes",
]


class TraceWindow(NamedTuple):
    identity: str
    path: pathlib.Path
    sha256: str


class Experiment(NamedTuple):
    policy: str
    offered_qps: float
    trace_window: TraceWindow
    repetition: int


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_trace_window(identity, path):
    path = pathlib.Path(path).resolve(strict=True)
    if not TRACE_WINDOW_ID.fullmatch(identity):
        raise ValueError(
            "trace-window identity must match [A-Za-z0-9][A-Za-z0-9._-]*")
    return TraceWindow(identity, path, sha256_file(path))


def generate_matrix(policies, offered_qps, trace_windows, repetitions):
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    unknown = set(policies) - set(POLICY_FILES)
    if unknown:
        raise ValueError(f"unknown policies: {', '.join(sorted(unknown))}")
    if not policies or not offered_qps or not trace_windows:
        raise ValueError("policies, offered QPS, and trace windows are required")
    if any(not math.isfinite(qps) or qps <= 0 for qps in offered_qps):
        raise ValueError("offered QPS must be finite and positive")
    if len({window.identity for window in trace_windows}) != len(trace_windows):
        raise ValueError("trace-window identities must be unique")
    return [
        Experiment(policy, qps, window, repetition)
        for window in sorted(trace_windows, key=lambda item: item.identity)
        for qps in sorted(set(offered_qps))
        for policy in sorted(set(policies))
        for repetition in range(1, repetitions + 1)
    ]


def _controlled_signature(metadata):
    return {
        key: metadata[key]
        for key in ("model_sha256", "git_revision", "build", "capacity", "run")
    }


def aggregate(summaries):
    if not summaries:
        raise ValueError("no summaries to aggregate")
    reference = _controlled_signature(summaries[0]["experiment"])
    trace_identities = {}
    policy_builds = {}
    observed = set()
    policies = set()
    comparison_cells = set()
    rows = []
    for summary in summaries:
        metadata = summary["experiment"]
        if _controlled_signature(metadata) != reference:
            raise ValueError("controlled configuration differs between runs")

        window = metadata["trace_window"]
        trace_fingerprint = (window["path"], window["sha256"])
        prior_trace = trace_identities.setdefault(window["identity"], trace_fingerprint)
        if prior_trace != trace_fingerprint:
            raise ValueError("trace-window identity refers to different content")

        policy = metadata["policy"]
        policy_fingerprint = (
            metadata["policy_source_sha256"],
            metadata["policy_header_sha256"],
            metadata["policy_config_sha256"],
        )
        prior_policy = policy_builds.setdefault(policy, policy_fingerprint)
        if prior_policy != policy_fingerprint:
            raise ValueError("policy build differs between runs")

        run_key = (policy, metadata["offered_qps"], window["identity"],
                   metadata["repetition"])
        if run_key in observed:
            raise ValueError("duplicate experiment run")
        observed.add(run_key)
        policies.add(policy)
        comparison_cells.add((metadata["offered_qps"], window["identity"],
                              metadata["repetition"]))

        scheduler = summary["scheduler"]
        rows.append({
            "policy": policy,
            "offered_qps": metadata["offered_qps"],
            "trace_window": window["identity"],
            "repetition": metadata["repetition"],
            "achieved_qps": summary["achieved_request_qps"],
            "ttft_p50_ns": summary["ttft"]["p50_ns"],
            "ttft_p95_ns": summary["ttft"]["p95_ns"],
            "ttft_p99_ns": summary["ttft"]["p99_ns"],
            "tpot_p99_ns": summary["tpot"]["p99_ns"],
            "peak_queue_depth": summary["peak_queued_requests"],
            "jain_fairness": summary["jain_fairness"],
            "scheduler_decision_count": scheduler["decision_count"],
            "scheduler_mean_ns": scheduler["mean_decision_time_ns"],
            "scheduler_p95_ns": scheduler["p95_decision_time_ns"],
            "scheduler_p99_ns": scheduler["p99_decision_time_ns"],
            "peak_rss_bytes": summary["peak_resident_memory_bytes"],
        })

    for cell in comparison_cells:
        present = {key[0] for key in observed if key[1:] == cell}
        if present != policies:
            raise ValueError("incomplete comparison matrix")
    return sorted(rows, key=lambda row: (
        row["trace_window"], row["offered_qps"], row["policy"],
        row["repetition"]))


def write_reports(rows, csv_path, markdown_path):
    csv_path = pathlib.Path(csv_path)
    markdown_path = pathlib.Path(markdown_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with markdown_path.open("w", encoding="utf-8") as stream:
        stream.write("| " + " | ".join(REPORT_COLUMNS) + " |\n")
        stream.write("| " + " | ".join("---" for _ in REPORT_COLUMNS) + " |\n")
        for row in rows:
            stream.write("| " + " | ".join(str(row[column]) for column in REPORT_COLUMNS)
                         + " |\n")


def _run(command, **kwargs):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def build_policy(policy, build_root, cc, cxx, jobs):
    stem = POLICY_FILES[policy]
    source = REPO_ROOT / "src" / "policies" / f"{stem}.cpp"
    header = REPO_ROOT / "src" / "policies" / f"{stem}.hpp"
    build_dir = build_root / policy
    command = [
        "cmake", "-S", REPO_ROOT, "-B", build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DQUICKSERVE_BENCHMARK_POLICY_SOURCE={source}",
        f"-DQUICKSERVE_BENCHMARK_POLICY_HEADER={header}",
    ]
    if policy == "proxqp":
        command.append("-DQUICKSERVE_BUILD_PROXQP_POLICY=ON")
        prefix = os.environ.get("PROXSUITE_PREFIX")
        if not prefix and shutil.which("brew"):
            prefix = subprocess.run(
                ["brew", "--prefix", "proxsuite"], check=True, text=True,
                capture_output=True).stdout.strip()
        if prefix:
            command.append(f"-DCMAKE_PREFIX_PATH={prefix}")
    environment = dict(os.environ, CC=cc, CXX=cxx)
    _run(command, env=environment)
    _run(["cmake", "--build", build_dir, "--target", "quickserve_benchmark",
          "-j", str(jobs)])
    return build_dir / "quickserve_benchmark"


def git_revision():
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        text=True, capture_output=True).stdout
    return revision + ("-dirty" if dirty else "")


def run_experiment(experiment, benchmark, args, model_hash, revision):
    run_dir = (args.output_dir / "runs" / experiment.trace_window.identity /
               f"qps-{experiment.offered_qps:g}" / experiment.policy /
               f"rep-{experiment.repetition:02d}")
    if run_dir.exists():
        raise RuntimeError(f"run output already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        benchmark, "--trace", experiment.trace_window.path,
        "--model", args.model, "--target-qps", f"{experiment.offered_qps:.12g}",
        "--max-requests", str(args.requests), "--output-mode", args.output_mode,
        "--output-dir", run_dir, "--context-size", str(args.context_size),
        "--batch-capacity", str(args.batch_capacity),
        "--max-sequences", str(args.max_sequences),
        "--token-budget", str(args.token_budget),
    ]
    config_hash = None
    if experiment.policy == "proxqp":
        if args.proxqp_config is None:
            raise ValueError("--proxqp-config is required when running ProxQP")
        config_hash = sha256_file(args.proxqp_config)
        command.extend(["--policy-config", args.proxqp_config])
    _run(command)
    summary_path = run_dir / "summary.json"
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    identity = summary["policy_identity"]
    summary["experiment"] = {
        "policy": experiment.policy,
        "offered_qps": experiment.offered_qps,
        "trace_window": {
            "identity": experiment.trace_window.identity,
            "path": str(experiment.trace_window.path),
            "sha256": experiment.trace_window.sha256,
        },
        "repetition": experiment.repetition,
        "model_sha256": model_hash,
        "git_revision": revision,
        "policy_source_sha256": identity["source_sha256"],
        "policy_header_sha256": identity["header_sha256"],
        "policy_config_sha256": config_hash,
        "build": summary["build"],
        "capacity": summary["environment"],
        "run": {"max_requests": args.requests, "output_mode": args.output_mode},
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def parse_trace_window(text):
    identity, separator, path = text.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected ID=PATH")
    try:
        return make_trace_window(identity, path)
    except (OSError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--trace-window", action="append", type=parse_trace_window,
                        required=True, metavar="ID=PATH")
    parser.add_argument("--qps", action="append", type=float, required=True)
    parser.add_argument("--policy", action="append", choices=sorted(POLICY_FILES))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--build-root", type=pathlib.Path,
                        default=pathlib.Path("/private/tmp/quickserve-policy-comparison"))
    parser.add_argument("--output-mode", choices=("natural", "trace-exact"),
                        default="trace-exact")
    parser.add_argument("--context-size", type=int, default=16384)
    parser.add_argument("--batch-capacity", type=int, default=512)
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--proxqp-config", type=pathlib.Path)
    parser.add_argument("--cc", default=os.environ.get("QUICKSERVE_CC", "/usr/bin/clang"))
    parser.add_argument("--cxx", default=os.environ.get("QUICKSERVE_CXX", "/usr/bin/clang++"))
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    return parser.parse_args(argv)


def validate_runtime_args(args):
    for name in ("requests", "jobs", "context_size", "batch_capacity",
                 "max_sequences", "token_budget"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")


def main(argv=None):
    args = parse_args(argv)
    try:
        args.model = args.model.resolve(strict=True)
        args.output_dir = args.output_dir.resolve()
        if args.output_dir.exists():
            raise ValueError(f"output directory already exists: {args.output_dir}")
        validate_runtime_args(args)
        policies = args.policy or list(POLICY_FILES)
        matrix = generate_matrix(policies, args.qps, args.trace_window,
                                 args.repetitions)
        if "proxqp" in policies and args.proxqp_config is None:
            raise ValueError("--proxqp-config is required when running ProxQP")
        if args.proxqp_config is not None:
            args.proxqp_config = args.proxqp_config.resolve(strict=True)
        model_hash = sha256_file(args.model)
        revision = git_revision()
        benchmarks = {
            policy: build_policy(policy, args.build_root, args.cc, args.cxx, args.jobs)
            for policy in sorted(set(policies))
        }
        summaries = [
            run_experiment(experiment, benchmarks[experiment.policy], args,
                           model_hash, revision)
            for experiment in matrix
        ]
        rows = aggregate(summaries)
        write_reports(rows, args.output_dir / "comparison.csv",
                      args.output_dir / "comparison.md")
    except (KeyError, OSError, ValueError, RuntimeError,
            subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
