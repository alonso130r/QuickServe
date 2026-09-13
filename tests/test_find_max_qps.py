import importlib.util
import pathlib
import sys
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "find_max_qps.py"


def load_script():
    spec = importlib.util.spec_from_file_location("find_max_qps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SearchTests(unittest.TestCase):
    def result(self, module, qps, sustainable):
        return module.ProbeResult(
            qps=qps, achieved_qps=qps,
            ttft_p99_ns=1_000_000_000, tpot_p99_ns=100_000_000,
            peak_queued=0, final_active=0, final_queued=0,
            failed=0, rejected=0, sustainable=sustainable)

    def test_default_sequence_capacity_is_sixteen(self):
        module = load_script()
        with mock.patch.object(sys, "argv", ["find_max_qps.py", "--model", "model.gguf"]):
            args = module.parse_args()
        self.assertEqual(args.max_sequences, 16)

    def test_sustainability_defaults_match_benchmark_slos(self):
        module = load_script()
        with mock.patch.object(sys, "argv", ["find_max_qps.py", "--model", "model.gguf"]):
            args = module.parse_args()
        self.assertEqual(args.throughput_ratio, 0.98)
        self.assertEqual(args.max_p99_ttft_ms, 2000)
        self.assertEqual(args.max_p99_tpot_ms, 200)

    def test_accepts_and_forwards_policy_config(self):
        module = load_script()
        with mock.patch.object(
                sys, "argv",
                ["find_max_qps.py", "--model", "model.gguf",
                 "--policy-config", "proxqp.conf"]):
            args = module.parse_args()

        command = module.build_probe_command(
            pathlib.Path("benchmark"), pathlib.Path("trace.qst"),
            pathlib.Path("model.gguf"), pathlib.Path("result"), 3.5, args)

        self.assertEqual(args.policy_config, pathlib.Path("proxqp.conf"))
        self.assertEqual(command[-2:], ["--policy-config", "proxqp.conf"])

    def test_omits_policy_config_when_not_requested(self):
        module = load_script()
        with mock.patch.object(sys, "argv",
                               ["find_max_qps.py", "--model", "model.gguf"]):
            args = module.parse_args()

        command = module.build_probe_command(
            pathlib.Path("benchmark"), pathlib.Path("trace.qst"),
            pathlib.Path("model.gguf"), pathlib.Path("result"), 3.5, args)

        self.assertNotIn("--policy-config", command)

    def test_exponential_sweep_then_binary_search(self):
        module = load_script()
        probes = []

        def probe(qps):
            probes.append(qps)
            result = self.result(module, qps, qps <= 10)
            return result._replace(achieved_qps=qps if qps <= 10 else 8.0)

        result = module.find_max_qps(probe, start_qps=2, max_qps=64, binary_steps=3)

        self.assertEqual(probes[:4], [2, 4, 8, 16])
        self.assertEqual(probes[4:], [12, 10, 11])
        self.assertEqual(result.qps, 10)
        self.assertTrue(result.sustainable)

    def test_stops_at_configured_max_when_every_probe_passes(self):
        module = load_script()

        result = module.find_max_qps(
            lambda qps: self.result(module, qps, True),
            start_qps=3,
            max_qps=10,
            binary_steps=4,
        )

        self.assertEqual(result.qps, 10)

    def test_requires_a_sustainable_starting_point(self):
        module = load_script()

        with self.assertRaisesRegex(RuntimeError, "starting QPS"):
            module.find_max_qps(
                lambda qps: self.result(module, qps, False),
                start_qps=2,
                max_qps=16,
                binary_steps=2,
            )

    def sustainable_metrics(self):
        return dict(target_qps=10, achieved_qps=10,
                    ttft_p99_ns=2_000_000_000, tpot_p99_ns=200_000_000,
                    peak_queued=25, final_active=0, final_queued=0,
                    failed=0, rejected=0, requests=100,
                    throughput_ratio=0.98, queue_fraction=0.25,
                    max_p99_ttft_ns=2_000_000_000,
                    max_p99_tpot_ns=200_000_000)

    def test_classification_requires_throughput(self):
        module = load_script()
        metrics = self.sustainable_metrics()
        self.assertTrue(module.is_sustainable(**metrics))
        self.assertFalse(module.is_sustainable(**{**metrics, "achieved_qps": 9.79}))

    def test_classification_requires_ttft_and_tpot_slos(self):
        module = load_script()
        metrics = self.sustainable_metrics()
        self.assertFalse(module.is_sustainable(**{**metrics, "ttft_p99_ns": 2_000_000_001}))
        self.assertFalse(module.is_sustainable(**{**metrics, "tpot_p99_ns": 200_000_001}))

    def test_classification_requires_bounded_and_drained_queue(self):
        module = load_script()
        metrics = self.sustainable_metrics()
        self.assertFalse(module.is_sustainable(**{**metrics, "peak_queued": 26}))
        self.assertFalse(module.is_sustainable(**{**metrics, "final_active": 1}))
        self.assertFalse(module.is_sustainable(**{**metrics, "final_queued": 1}))

    def test_classification_rejects_failures_and_rejections(self):
        module = load_script()
        metrics = self.sustainable_metrics()
        self.assertFalse(module.is_sustainable(**{**metrics, "failed": 1}))
        self.assertFalse(module.is_sustainable(**{**metrics, "rejected": 1}))

    def test_reports_each_failed_sustainability_gate(self):
        module = load_script()
        metrics = self.sustainable_metrics()
        failures = module.sustainability_failures(**{
            **metrics,
            "achieved_qps": 9.7,
            "ttft_p99_ns": 5_736_041_237,
            "peak_queued": 26,
            "failed": 1,
        })

        self.assertEqual(failures, [
            "achieved QPS 9.700 < 9.800",
            "p99 TTFT 5736.0 ms > 2000.0 ms",
            "peak queue 26 > 25",
            "failed requests 1 > 0",
        ])

    def test_validates_latency_limits(self):
        module = load_script()
        with mock.patch.object(sys, "argv", ["find_max_qps.py", "--model", "model.gguf",
                                             "--max-p99-ttft-ms", "0"]):
            with self.assertRaisesRegex(ValueError, "TTFT"):
                module.validate_args(module.parse_args())
        with mock.patch.object(sys, "argv", ["find_max_qps.py", "--model", "model.gguf",
                                             "--max-p99-tpot-ms", "nan"]):
            with self.assertRaisesRegex(ValueError, "TPOT"):
                module.validate_args(module.parse_args())

    def test_builds_probe_result_from_summary_metrics(self):
        module = load_script()
        with mock.patch.object(sys, "argv", ["find_max_qps.py", "--model", "model.gguf"]):
            args = module.parse_args()
        summary = {
            "achieved_request_qps": 9.9,
            "ttft": {"p99_ns": 1_500_000_000},
            "tpot": {"p99_ns": 150_000_000},
            "counts": {"failed": 0, "rejected": 0},
            "peak_queued_requests": 4,
            "final_active_requests": 0,
            "final_queued_requests": 0,
        }

        result = module.probe_result_from_summary(10, summary, args)

        self.assertEqual(result.ttft_p99_ns, 1_500_000_000)
        self.assertEqual(result.tpot_p99_ns, 150_000_000)
        self.assertTrue(result.sustainable)

    def test_rejects_malformed_summary_metrics(self):
        module = load_script()
        with mock.patch.object(sys, "argv", ["find_max_qps.py", "--model", "model.gguf"]):
            args = module.parse_args()
        valid = {
            "achieved_request_qps": 9.9,
            "ttft": {"p99_ns": 1_500_000_000},
            "tpot": {"p99_ns": 150_000_000},
            "counts": {"failed": 0, "rejected": 0},
            "peak_queued_requests": 4,
            "final_active_requests": 0,
            "final_queued_requests": 0,
        }
        malformed = [
            {**valid, "achieved_request_qps": float("nan")},
            {**valid, "ttft": {"p99_ns": -1}},
            {**valid, "tpot": {"p99_ns": float("inf")}},
            {**valid, "ttft": {"p99_ns": 1.5}},
            {**valid, "tpot": {"p99_ns": 1.5}},
            {**valid, "peak_queued_requests": 1.5},
            {**valid, "final_active_requests": True},
            {**valid, "counts": {"failed": -1, "rejected": 0}},
        ]

        for summary in malformed:
            with self.subTest(summary=summary):
                with self.assertRaisesRegex(RuntimeError, "invalid sustainability metrics"):
                    module.probe_result_from_summary(10, summary, args)


if __name__ == "__main__":
    unittest.main()
