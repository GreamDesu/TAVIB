"""Small checks for the public loading example and dataset-independent commands."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.load_recordings import EXAMPLES, load_examples


ROOT = Path(__file__).resolve().parents[1]


class LoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.expected = {}
        for name, (relative, header, columns) in EXAMPLES.items():
            path = self.root / "data" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            values = np.arange(3 * columns, dtype=float).reshape(3, columns)
            self.expected[name] = values
            np.savetxt(path, values, delimiter=",", comments="",
                       header=",".join(["duplicate header"] * columns) if header else "")

    def test_preserves_positional_columns_and_samples(self):
        actual = load_examples(self.root)
        for name in self.expected:
            np.testing.assert_array_equal(actual[name], self.expected[name])

    def test_rejects_wrong_column_count(self):
        path = self.root / "data" / EXAMPLES["slippage"][0]
        np.savetxt(path, np.zeros((3, 7)), delimiter=",")
        with self.assertRaisesRegex(ValueError, "8 columns"):
            load_examples(self.root)

    def test_rejects_nonfinite_values(self):
        path = self.root / "data" / EXAMPLES["slippage"][0]
        values = np.zeros((3, 8))
        values[0, 0] = np.nan
        np.savetxt(path, values, delimiter=",")
        with self.assertRaisesRegex(ValueError, "finite data"):
            load_examples(self.root)


class CommandTests(unittest.TestCase):
    def test_help_from_an_unrelated_working_directory(self):
        for script in ["examples/load_recordings.py", "tactile_dataset_validation/run_all.py",
                       "tactile_dataset_validation/validation_slippage/run.py"]:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as cwd:
                result = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                                        cwd=cwd, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
