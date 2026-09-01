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
    def test_default_sequence_capacity_is_sixteen(self):
        module = load_script()
        with mock.patch.object(sys, "argv", ["find_max_qps.py", "--model", "model.gguf"]):
            args = module.parse_args()
        self.assertEqual(args.max_sequences, 16)

    def test_exponential_sweep_then_binary_search(self):
        module = load_script()
        probes = []

        def probe(qps):
            probes.append(qps)
            return module.ProbeResult(qps, qps if qps <= 10 else 8.0, 0, qps <= 10)

        result = module.find_max_qps(probe, start_qps=2, max_qps=64, binary_steps=3)

        self.assertEqual(probes[:4], [2, 4, 8, 16])
        self.assertEqual(probes[4:], [12, 10, 11])
        self.assertEqual(result.qps, 10)
        self.assertTrue(result.sustainable)

    def test_stops_at_configured_max_when_every_probe_passes(self):
        module = load_script()

        result = module.find_max_qps(
            lambda qps: module.ProbeResult(qps, qps, 0, True),
            start_qps=3,
            max_qps=10,
            binary_steps=4,
        )

        self.assertEqual(result.qps, 10)

    def test_requires_a_sustainable_starting_point(self):
        module = load_script()

        with self.assertRaisesRegex(RuntimeError, "starting QPS"):
            module.find_max_qps(
                lambda qps: module.ProbeResult(qps, 0.1, 20, False),
                start_qps=2,
                max_qps=16,
                binary_steps=2,
            )

    def test_classification_uses_throughput_and_queue_limits(self):
        module = load_script()

        self.assertTrue(module.is_sustainable(10, 9.2, 20, 100, 0.9, 0.25))
        self.assertFalse(module.is_sustainable(10, 8.9, 0, 100, 0.9, 0.25))
        self.assertFalse(module.is_sustainable(10, 10, 26, 100, 0.9, 0.25))


if __name__ == "__main__":
    unittest.main()
