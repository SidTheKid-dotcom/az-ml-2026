"""Queue keeps going after a failure and runs one GPU job at a time."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "run_gpu_queue.py"


class GpuQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = self.root / "logs"

    def run_queue(self, jobs):
        path = self.root / "jobs.json"
        path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
        smi = self.root / "nvidia-smi"
        smi.write_text("#!/usr/bin/env python3\nprint('46068, 0, 45498')\n", encoding="utf-8")
        smi.chmod(0o700)
        return subprocess.run(
            [sys.executable, str(RUNNER), "--queue", str(path), "--logs", str(self.logs),
             "--nvidia-smi", str(smi), "--poll-seconds", "0.05"],
            capture_output=True, text=True, timeout=20,
        )

    def report(self):
        return json.loads(sorted(self.logs.glob("queue-*.json"))[-1].read_text())

    def test_failure_does_not_cancel_the_next_job(self):
        result = self.run_queue([
            {"name": "dies", "kind": "cpu", "threads": 1, "vram_mib": 0,
             "command": [sys.executable, "-c", "raise SystemExit(1)"]},
            {"name": "lives", "kind": "cpu", "threads": 1, "vram_mib": 0,
             "command": [sys.executable, "-c", "print('kept going')"]},
        ])
        self.assertEqual(result.returncode, 1, result.stderr)
        jobs = {row["name"]: row["status"] for row in self.report()["jobs"]}
        self.assertEqual(jobs["dies"], "failed")
        self.assertEqual(jobs["lives"], "complete")

    def test_one_gpu_job_at_a_time(self):
        result = self.run_queue([
            {"name": f"gpu-{i}", "kind": "gpu", "threads": 1, "vram_mib": 8000,
             "command": [sys.executable, "-c", "import time; time.sleep(0.2)"]}
            for i in range(2)
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row["active_gpu_at_launch"] for row in self.report()["jobs"]], [1, 1])


if __name__ == "__main__":
    unittest.main()
