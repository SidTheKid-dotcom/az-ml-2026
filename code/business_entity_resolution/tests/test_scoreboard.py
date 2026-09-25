"""A partial report checkout must never erase the historical experiment ledger."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ber import scoreboard


class ScoreboardPreservationTests(unittest.TestCase):
    def test_missing_completed_report_refuses_to_replace_existing_ledger(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "artifacts" / "quick_v1" / "results.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(json.dumps({"results": [{"model": "example", "tune": {
                "macro_f05": .9, "micro_precision": .9, "micro_recall": .9,
                "threshold": .5, "singleton_accuracy": 1.0,
                "reference_count": 2, "predicted_links": 2,
            }}]}))
            output = root / "learnings"
            argv = ["scoreboard", "--artifacts", str(root / "artifacts"), "--output", str(output)]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                scoreboard.main()
            originals = {name: (output / name).read_bytes() for name in
                         ("experiments.json", "experiments.csv", "scoreboard.md")}
            report_path.unlink()
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "Refusing"):
                scoreboard.main()
            for name, original in originals.items():
                self.assertEqual((output / name).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
