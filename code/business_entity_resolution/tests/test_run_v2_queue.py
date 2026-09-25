"""Bounded queue smoke: actual subprocesses, fake VRAM readings, and real file locks."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "run_v2_queue.py"


class V2QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = self.root / "logs"

    def run_queue(self, *args):
        return subprocess.run([sys.executable, str(RUNNER), "--logs", str(self.logs),
                               "--poll-seconds", "0.05", *map(str, args)],
                              capture_output=True, text=True, timeout=15)

    def latest_report(self):
        return json.loads(sorted(self.logs.glob("queue-*.json"))[-1].read_text())

    def fake_gpu(self, used):
        path = self.root / "fake-nvidia-smi"
        path.write_text(f"#!/usr/bin/env python3\nprint('46068, {used}, {45498 - used}')\n")
        path.chmod(0o700)
        return path

    def test_smoke_logs_and_skips_successful_repeat(self):
        first = self.run_queue("--smoke-cpu")
        self.assertEqual(first.returncode, 0, first.stderr)
        report = self.latest_report()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["jobs"][0]["status"], "complete")
        self.assertIn("cpu_budget_threads", report["cgroup"])
        self.assertIn("memory_max_bytes", report["cgroup"])
        self.assertIn("v2 queue smoke ok", Path(report["jobs"][0]["log"]).read_text())

        second = self.run_queue("--smoke-cpu")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(self.latest_report()["jobs"], [{"name": "p0-cpu-smoke", "status": "already_complete"}])
        self.assertEqual(len(list(self.logs.glob("*.log"))), 1)

    def test_another_runner_cannot_start_with_same_logs(self):
        self.logs.mkdir()
        with (self.logs / "queue.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_queue("--smoke-cpu")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Another v2 queue", result.stderr)
        self.assertFalse(list(self.logs.glob("*.log")))

    def test_gpu_preflight_blocks_over_cap_then_bounded_smoke_passes(self):
        smi = self.fake_gpu(38000)
        blocked = self.run_queue("--smoke-gpu", "--nvidia-smi", smi)
        self.assertEqual(blocked.returncode, 1, blocked.stderr)
        self.assertEqual(self.latest_report()["jobs"], [{"name": "p0-gpu-smoke", "status": "not_started"}])

        smi = self.fake_gpu(0)
        passed = self.run_queue("--smoke-gpu", "--nvidia-smi", smi)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        report = self.latest_report()
        self.assertEqual(report["vram_cap_mib"], 38146)
        self.assertEqual(report["jobs"][0]["status"], "complete")
        self.assertIn("v2 queue smoke ok", Path(report["jobs"][0]["log"]).read_text())

    def test_concurrent_slots_and_reservations_never_exceed_caps(self):
        smi = self.fake_gpu(0)
        for size, prefix in [(5000, "slots"), (15000, "reservation")]:
            jobs = [{"name": f"{prefix}-gpu-{i}", "kind": "gpu", "vram_mib": size,
                     "threads": 1, "command": [sys.executable, "-c", "import time; time.sleep(.15)"]}
                    for i in range(4)]
            jobs.extend({"name": f"{prefix}-cpu-{i}", "kind": "cpu", "threads": 1,
                         "command": [sys.executable, "-c", "import time; time.sleep(.15)"]}
                        for i in range(2))
            manifest = self.root / f"{prefix}.json"
            manifest.write_text(json.dumps({"jobs": jobs}))
            result = self.run_queue("--queue", manifest, "--nvidia-smi", smi)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = self.latest_report()
            self.assertEqual(len(report["jobs"]), 6)
            self.assertTrue(all(job["status"] == "complete" for job in report["jobs"]))
            self.assertTrue(all(job["active_gpu_at_launch"] <= 3 and job["active_cpu_at_launch"] <= 1
                                and job["active_threads_at_launch"] <= 6
                                and job["reserved_vram_at_launch_mib"] <= 38146 for job in report["jobs"]))
            if size == 15000:
                self.assertTrue(all(job["reserved_vram_at_launch_mib"] <= 30000 for job in report["jobs"]))


if __name__ == "__main__":
    unittest.main()
