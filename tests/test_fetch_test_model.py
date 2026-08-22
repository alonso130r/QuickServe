import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "fetch_test_model.py"


def load_fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_test_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SelectModelFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fetch = load_fetch_module()

    def test_selects_exact_basename(self):
        files = [
            "README.md",
            "models/Qwen3.5-2B-Q4_K_M.gguf",
            "Qwen3.5-2B-Q8_0.gguf",
        ]

        self.assertEqual(
            self.fetch.select_model_file(files, "Q4_K_M"),
            "models/Qwen3.5-2B-Q4_K_M.gguf",
        )

    def test_excludes_near_collisions(self):
        files = [
            "Qwen3.5-2B-UD-Q4_K_M.gguf",
            "Qwen3.5-2B-Q4_K_M_EXTRA.gguf",
            "nested/Qwen3.5-2B-Q4_K_M.gguf",
        ]

        self.assertEqual(
            self.fetch.select_model_file(files, "Q4_K_M"),
            "nested/Qwen3.5-2B-Q4_K_M.gguf",
        )

    def test_rejects_missing_match(self):
        with self.assertRaisesRegex(ValueError, "No file named"):
            self.fetch.select_model_file(["Qwen3.5-2B-Q8_0.gguf"], "Q4_K_M")

    def test_rejects_duplicate_matching_paths(self):
        files = [
            "a/Qwen3.5-2B-Q4_K_M.gguf",
            "b/Qwen3.5-2B-Q4_K_M.gguf",
        ]

        with self.assertRaisesRegex(ValueError, "multiple files named"):
            self.fetch.select_model_file(files, "Q4_K_M")


if __name__ == "__main__":
    unittest.main()
