import contextlib
import io
import runpy
import unittest
from pathlib import Path

from graphspace import cli

ROOT = Path(__file__).resolve().parent.parent


class TestCliAndExamples(unittest.TestCase):

    def test_cli_demo_runs(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(), 0)
        self.assertIn("graph=demo_add", output.getvalue())

    def test_example_runs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT / "examples" / "demo.py"), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
