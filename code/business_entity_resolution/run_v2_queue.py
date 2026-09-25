#!/usr/bin/env python3
"""Small v2 command queue for the single-GPU pod; no v1 paths are implicit.

Usage: python run_v2_queue.py --queue /path/to/jobs.json
Manifest: {"jobs": [{"name": "unique-name", "kind": "gpu", "threads": 2,
                     "vram_mib": 10000, "command": ["python", "-m", "..."],
                     "cwd": "/tmp/az-ml-v2", "timeout_seconds": 3600}]}
CPU jobs omit vram_mib. Commands are argv lists, never shell expressions. Keep
one logs directory for all queues on the pod so its lock covers every runner.
Use --smoke-gpu for one bounded, GPU-preflight-only command (no model run).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


# nvidia-smi reports MiB; 40 decimal GB is 38,146 MiB when rounded down.
VRAM_CAP_MIB = 40_000_000_000 // (1024 * 1024)
MAX_GPU_JOBS = 3
MAX_CPU_JOBS = 1
POD_CPU_BUDGET = 6
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cgroup_value(*paths: str) -> str | None:
    for name in paths:
        path = Path(name)
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return None


def cgroup_limits() -> dict:
    cpu_max = cgroup_value("/sys/fs/cgroup/cpu.max")
    if cpu_max and cpu_max.split()[0] != "max":
        quota, period = map(int, cpu_max.split())
    else:
        quota = int(cgroup_value("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") or -1)
        period = int(cgroup_value("/sys/fs/cgroup/cpu/cpu.cfs_period_us") or 100000)
    cores = quota / period if quota > 0 and period > 0 else None
    raw_max = cgroup_value("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
    memory_max = int(raw_max) if raw_max and raw_max != "max" and int(raw_max) < 1 << 60 else None
    raw_current = cgroup_value("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    budget = min(POD_CPU_BUDGET, affinity, max(1, int(cores)) if cores else affinity)
    return {"cpu_quota_cores": cores, "cpu_affinity": affinity, "cpu_budget_threads": budget,
            "memory_max_bytes": memory_max, "memory_current_bytes": int(raw_current) if raw_current else None}


@dataclass(frozen=True)
class GPU:
    total_mib: int
    used_mib: int
    free_mib: int


def gpu_memory(nvidia_smi: str) -> GPU:
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"nvidia-smi VRAM preflight failed: {exc}") from exc
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("Queue expects exactly one visible GPU")
    try:
        gpu = GPU(*(int(value.strip()) for value in lines[0].split(",")))
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid nvidia-smi memory response") from exc
    if min(gpu.total_mib, gpu.free_mib) <= 0 or gpu.used_mib < 0:
        raise ValueError("nvidia-smi returned unusable VRAM")
    return gpu


@dataclass(frozen=True)
class Job:
    name: str
    kind: str
    command: tuple[str, ...]
    cwd: str
    threads: int
    vram_mib: int
    timeout_seconds: float | None

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def jobs_from_manifest(path: Path) -> list[Job]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list) or not data["jobs"]:
        raise ValueError("Queue manifest must contain a nonempty jobs list")
    jobs = []
    for item in data["jobs"]:
        if not isinstance(item, dict):
            raise ValueError("Each job must be an object")
        name, kind = item.get("name"), item.get("kind")
        command = item.get("command")
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError(f"Unsafe or missing job name: {name!r}")
        if kind not in ("gpu", "cpu") or not isinstance(command, list) or not command or not all(
            isinstance(arg, str) and arg for arg in command
        ):
            raise ValueError(f"{name}: kind must be gpu/cpu and command must be a nonempty argv list")
        threads = item.get("threads")
        vram = item.get("vram_mib", 0)
        if type(threads) is not int or not 1 <= threads <= POD_CPU_BUDGET:
            raise ValueError(f"{name}: threads must be an integer from 1 to {POD_CPU_BUDGET}")
        if type(vram) is not int or (kind == "gpu" and not 0 < vram <= VRAM_CAP_MIB) or (kind == "cpu" and vram != 0):
            raise ValueError(f"{name}: GPU jobs require 1..{VRAM_CAP_MIB} vram_mib; CPU jobs require 0")
        timeout = item.get("timeout_seconds")
        if timeout is not None and (type(timeout) not in (int, float) or timeout <= 0):
            raise ValueError(f"{name}: timeout_seconds must be positive")
        cwd = Path(item.get("cwd", Path.cwd()))
        if not cwd.is_absolute():
            cwd = path.parent / cwd
        cwd = cwd.resolve()
        if not cwd.is_dir():
            raise ValueError(f"{name}: working directory does not exist: {cwd}")
        jobs.append(Job(name, kind, tuple(command), str(cwd), threads, vram, timeout))
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job names in queue")
    return jobs


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def execute(jobs: list[Job], logs: Path, nvidia_smi: str, poll_seconds: float) -> int:
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another v2 queue is already running with logs at {logs}") from exc
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-{os.getpid()}"
        complete = logs / "complete"
        complete.mkdir(exist_ok=True)
        resources = cgroup_limits()
        initial_gpu = gpu_memory(nvidia_smi) if any(job.kind == "gpu" for job in jobs) else None
        cap = min(VRAM_CAP_MIB, initial_gpu.total_mib - 2048) if initial_gpu else None
        if cap is not None and cap <= 0:
            raise ValueError("GPU has insufficient usable VRAM")
        print(f"cgroup: {resources}; GPU: {initial_gpu}; VRAM scheduling cap: {cap} MiB", flush=True)
        report = {"run_id": run_id, "started_utc": utc_now(), "cgroup": resources,
                  "gpu_initial": asdict(initial_gpu) if initial_gpu else None, "vram_cap_mib": cap,
                  "max_gpu_jobs": MAX_GPU_JOBS, "max_cpu_jobs": MAX_CPU_JOBS, "jobs": []}
        pending = []
        for job in jobs:
            marker = complete / f"{job.name}.json"
            if marker.exists():
                previous = json.loads(marker.read_text(encoding="utf-8"))
                if previous.get("fingerprint") != job.fingerprint():
                    raise ValueError(f"{job.name}: completed name belongs to a different command; choose a new name")
                report["jobs"].append({"name": job.name, "status": "already_complete"})
            else:
                pending.append(job)
        running: dict[str, tuple[Job, subprocess.Popen, object, dict, float]] = {}
        failed = False
        try:
            while pending or running:
                for name, (job, proc, output, entry, started) in list(running.items()):
                    code = proc.poll()
                    timed_out = bool(code is None and job.timeout_seconds and
                                     time.monotonic() - started >= job.timeout_seconds)
                    if timed_out:
                        terminate(proc)
                        code = proc.poll()
                    if code is None:
                        continue
                    output.close()
                    del running[name]
                    entry.update(status="timed_out" if timed_out else "complete" if code == 0 else "failed",
                                 exit_code=code, elapsed_seconds=round(time.monotonic() - started, 3),
                                 finished_utc=utc_now())
                    if code == 0 and not timed_out:
                        atomic_json(complete / f"{name}.json", {"fingerprint": job.fingerprint(), "run_id": run_id,
                                                             "finished_utc": entry["finished_utc"]})
                    else:
                        failed = True
                if not failed:
                    for job in pending[:]:
                        active = [record[0] for record in running.values()]
                        if sum(other.threads for other in active) + job.threads > resources["cpu_budget_threads"]:
                            continue
                        if sum(other.kind == job.kind for other in active) >= (MAX_GPU_JOBS if job.kind == "gpu" else MAX_CPU_JOBS):
                            continue
                        if job.kind == "gpu":
                            gpu = gpu_memory(nvidia_smi)  # fresh reading immediately before every GPU launch
                            reserved = sum(other.vram_mib for other in active if other.kind == "gpu")
                            # Own jobs may not have allocated their full reservation yet; other GPU users
                            # may allocate concurrently. Check both nvidia-smi and the reserved upper bound.
                            if max(gpu.used_mib, initial_gpu.used_mib + reserved) + job.vram_mib > cap or gpu.free_mib < job.vram_mib:
                                continue
                        path = logs / f"{run_id}-{job.name}.log"
                        output = path.open("w", encoding="utf-8")
                        env = os.environ.copy()
                        env.update({name: str(job.threads) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
                        env["TOKENIZERS_PARALLELISM"] = "false"
                        try:
                            # A child retains the lock even if the supervisor dies unexpectedly.
                            proc = subprocess.Popen(job.command, cwd=job.cwd, env=env, stdout=output,
                                                    stderr=subprocess.STDOUT, start_new_session=True,
                                                    pass_fds=(lock.fileno(),))
                        except OSError:
                            output.close()
                            raise
                        entry = {"name": job.name, "kind": job.kind, "status": "running", "pid": proc.pid,
                                 "log": str(path), "threads": job.threads, "vram_reserved_mib": job.vram_mib,
                                 "active_gpu_at_launch": sum(other.kind == "gpu" for other in active) + (job.kind == "gpu"),
                                 "active_cpu_at_launch": sum(other.kind == "cpu" for other in active) + (job.kind == "cpu"),
                                 "active_threads_at_launch": sum(other.threads for other in active) + job.threads,
                                 "reserved_vram_at_launch_mib": sum(other.vram_mib for other in active) + job.vram_mib,
                                 "started_utc": utc_now()}
                        running[job.name] = (job, proc, output, entry, time.monotonic())
                        report["jobs"].append(entry)
                        pending.remove(job)
                        print(f"started {job.name} ({job.kind}, pid {proc.pid}) -> {path}", flush=True)
                if not running and pending:
                    if not failed:
                        failed = True
                        print("Queue blocked: no pending job fits the current GPU VRAM/CPU budget", file=sys.stderr)
                    report["jobs"].extend({"name": job.name, "status": "not_started"} for job in pending)
                    break
                if running:
                    time.sleep(poll_seconds)
        except BaseException:
            for _, proc, output, _, _ in running.values():
                terminate(proc)
                output.close()
            raise
        report["finished_utc"] = utc_now()
        report["status"] = "failed" if failed else "complete"
        report_path = logs / f"queue-{run_id}.json"
        atomic_json(report_path, report)
        print(f"queue {report['status']}: {report_path}", flush=True)
        return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queue", type=Path, help="JSON manifest of independent argv commands")
    source.add_argument("--smoke-cpu", action="store_true", help="run one bounded CPU-only command")
    source.add_argument("--smoke-gpu", action="store_true", help="run one bounded GPU-preflight command (no CUDA load)")
    parser.add_argument("--logs", type=Path, default=Path("/tmp/az-ml-v2/logs"))
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not 0.05 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 0.05 and 60")
    try:
        if args.queue:
            jobs = jobs_from_manifest(args.queue.resolve())
        else:
            kind = "gpu" if args.smoke_gpu else "cpu"
            jobs = [Job(f"p0-{kind}-smoke", kind,
                        (sys.executable, "-c", "print('v2 queue smoke ok')"), str(Path.cwd()), 1,
                        256 if kind == "gpu" else 0, 10)]
        return execute(jobs, args.logs.resolve(), args.nvidia_smi, args.poll_seconds)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"v2 queue: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
