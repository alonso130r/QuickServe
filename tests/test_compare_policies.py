#!/usr/bin/env python3

import importlib.util
import json
import pathlib
import tempfile
import types
import unittest


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "compare_policies.py"
SPEC = importlib.util.spec_from_file_location("compare_policies", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def summary(policy="fifo", qps=8.0, window="azure-a", repetition=1):
    return {
        "experiment": {
            "policy": policy,
            "offered_qps": qps,
            "trace_window": {
                "identity": window,
                "path": f"/traces/{window}.qst",
                "sha256": f"sha-{window}",
            },
            "repetition": repetition,
            "model_sha256": "model-sha",
            "git_revision": "abc123",
            "policy": policy,
            "policy_source_sha256": f"source-{policy}",
            "policy_header_sha256": f"header-{policy}",
            "policy_config_sha256": None,
            "build": {
                "build_type": "Release",
                "compiler_id": "AppleClang",
                "compiler_version": "17.0",
            },
            "capacity": {
                "context_size": 16384,
                "batch_capacity": 512,
                "max_sequences": 16,
                "token_budget": 512,
            },
            "run": {"max_requests": 32, "output_mode": "trace-exact"},
        },
        "achieved_request_qps": qps - 0.1,
        "ttft": {"p50_ns": 10, "p95_ns": 20, "p99_ns": 30},
        "tpot": {"p99_ns": 40},
        "peak_queued_requests": 5,
        "jain_fairness": 0.95,
        "scheduler": {
            "decision_count": 7,
            "mean_decision_time_ns": 11.5,
            "p95_decision_time_ns": 18,
            "p99_decision_time_ns": 21,
        },
        "peak_resident_memory_bytes": 123456,
    }


class MatrixTests(unittest.TestCase):
    def test_matrix_is_deterministic_and_uses_identical_loads(self):
        windows = [
            module.TraceWindow("b", pathlib.Path("/tmp/b.qst"), "hash-b"),
            module.TraceWindow("a", pathlib.Path("/tmp/a.qst"), "hash-a"),
        ]
        matrix = module.generate_matrix(
            ["proxqp", "fifo"], [12.0, 4.0], windows, 2)

        keys = [(x.policy, x.offered_qps, x.trace_window.identity, x.repetition)
                for x in matrix]
        self.assertEqual(keys, [
            (policy, qps, window, repetition)
            for window in ("a", "b")
            for qps in (4.0, 12.0)
            for policy in ("fifo", "proxqp")
            for repetition in (1, 2)
        ])

    def test_trace_window_identity_includes_file_hash(self):
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "window.qst"
            path.write_bytes(b"fixed trace records")
            first = module.make_trace_window("azure-01", path)
            second = module.make_trace_window("azure-01", path)

        self.assertEqual(first, second)
        self.assertEqual(first.identity, "azure-01")
        self.assertEqual(
            first.sha256,
            "43ccb8a9fb5c4b1b39b107352bfeb08860f5c8a54f116a4101b0742765452f1a",
        )

    def test_rejects_nonfinite_or_nonpositive_qps(self):
        window = module.TraceWindow("a", pathlib.Path("/tmp/a.qst"), "hash-a")
        for qps in (0, -1, float("inf"), float("-inf"), float("nan")):
            with self.subTest(qps=qps):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    module.generate_matrix(["fifo"], [qps], [window], 1)

    def test_rejects_unsafe_trace_window_identities(self):
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "window.qst"
            path.write_bytes(b"trace")
            for identity in ("", " ", ".", "..", "a/b", "a\\b", "a b",
                             "a:b", "a\0b", "a\nb"):
                with self.subTest(identity=identity):
                    with self.assertRaisesRegex(ValueError, "trace-window identity"):
                        module.make_trace_window(identity, path)

    def test_accepts_conservative_trace_window_identity_characters(self):
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "window.qst"
            path.write_bytes(b"trace")
            window = module.make_trace_window("azure_Code-01.2", path)
        self.assertEqual(window.identity, "azure_Code-01.2")

    def test_rejects_nonpositive_runtime_capacities(self):
        valid = {
            "requests": 4,
            "jobs": 1,
            "context_size": 4096,
            "batch_capacity": 512,
            "max_sequences": 16,
            "token_budget": 512,
        }
        for field in valid:
            with self.subTest(field=field):
                values = {**valid, field: 0}
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    module.validate_runtime_args(types.SimpleNamespace(**values))


class AggregationTests(unittest.TestCase):
    def test_aggregation_extracts_required_report_columns(self):
        records = [summary("fifo"), summary("continuous_fifo")]
        rows = module.aggregate(records)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["achieved_qps"], 7.9)
        self.assertEqual(rows[0]["ttft_p50_ns"], 10)
        self.assertEqual(rows[0]["ttft_p95_ns"], 20)
        self.assertEqual(rows[0]["ttft_p99_ns"], 30)
        self.assertEqual(rows[0]["tpot_p99_ns"], 40)
        self.assertEqual(rows[0]["peak_queue_depth"], 5)
        self.assertEqual(rows[0]["jain_fairness"], 0.95)
        self.assertEqual(rows[0]["scheduler_decision_count"], 7)
        self.assertEqual(rows[0]["scheduler_p99_ns"], 21)
        self.assertEqual(rows[0]["peak_rss_bytes"], 123456)

    def test_rejects_mismatched_controlled_configuration(self):
        fields = [
            ("model_sha256", "other-model"),
            ("git_revision", "other-revision"),
            ("build", {"build_type": "Debug", "compiler_id": "AppleClang",
                       "compiler_version": "17.0"}),
            ("capacity", {"context_size": 8192, "batch_capacity": 512,
                          "max_sequences": 16, "token_budget": 512}),
            ("run", {"max_requests": 64, "output_mode": "trace-exact"}),
        ]
        for field, value in fields:
            with self.subTest(field=field):
                other = summary("continuous_fifo")
                other["experiment"][field] = value
                with self.assertRaisesRegex(ValueError, "controlled configuration"):
                    module.aggregate([summary("fifo"), other])

    def test_rejects_trace_identity_reused_for_different_content(self):
        other = summary("continuous_fifo")
        other["experiment"]["trace_window"]["sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "trace-window identity"):
            module.aggregate([summary("fifo"), other])

    def test_rejects_incomplete_policy_pairing(self):
        records = [summary("fifo", window="a"),
                   summary("continuous_fifo", window="a"),
                   summary("fifo", window="b")]
        with self.assertRaisesRegex(ValueError, "incomplete comparison matrix"):
            module.aggregate(records)

    def test_writes_csv_and_markdown(self):
        rows = module.aggregate([summary("fifo"), summary("continuous_fifo")])
        with tempfile.TemporaryDirectory() as root:
            csv_path = pathlib.Path(root) / "comparison.csv"
            markdown_path = pathlib.Path(root) / "comparison.md"
            module.write_reports(rows, csv_path, markdown_path)
            csv_text = csv_path.read_text()
            markdown_text = markdown_path.read_text()

        self.assertIn("scheduler_mean_ns", csv_text)
        self.assertIn("peak_rss_bytes", csv_text)
        self.assertIn("| policy |", markdown_text)
        self.assertIn("continuous_fifo", markdown_text)


if __name__ == "__main__":
    unittest.main()
