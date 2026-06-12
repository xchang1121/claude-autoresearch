"""Local worker daemon lifecycle helpers."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from .worker_config import WorkerConfig

SCRIPTS_DIR = Path(__file__).resolve().parents[1]


def worker_log_path(port: int) -> str:
    if os.name == "posix":
        return f"/tmp/akg_worker_{port}.log"
    log_dir = Path(os.environ.get("AR_STATE_DIR", str(Path.home() / ".autoresearch_state"))) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir / f"worker_{port}.log")


def _env_float(key: str, default: float) -> float:
    try:
        value = float(os.environ.get(key, ""))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def curl_status(host: str, port: int, timeout: Optional[float] = None) -> Optional[dict]:
    if timeout is None:
        timeout = WorkerConfig.load(None).timing.status_timeout
    try:
        with urlopen(Request(f"http://{host}:{port}/api/v1/status", method="GET"), timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, socket.timeout, ConnectionError, Exception):
        return None


def curl_health(host: str, port: int, timeout: Optional[float] = None) -> Optional[dict]:
    if timeout is None:
        timeout = WorkerConfig.load(None).timing.status_timeout * 2
    try:
        with urlopen(Request(f"http://{host}:{port}/api/v1/health", method="GET"), timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, socket.timeout, ConnectionError, Exception):
        return None


def is_ready(status: Optional[dict]) -> bool:
    return isinstance(status, dict) and str(status.get("status", "")).lower() in ("ready", "ok")


def _find_pid_on_port(port: int) -> Optional[int]:
    if os.name == "posix":
        try:
            out = subprocess.run(
                ["lsof", "-iTCP:" + str(port), "-sTCP:LISTEN", "-t"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in out.stdout.splitlines():
                value = line.strip()
                if value.isdigit():
                    return int(value)
        except Exception:
            return None
        return None

    try:
        ps_cmd = (
            f"$c = Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; "
            "if ($c) { Write-Output $c.OwningProcess }"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
        value = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        return int(value) if value.isdigit() else None
    except Exception:
        return None


def _cmdline(pid: int) -> str:
    if os.name == "posix":
        try:
            return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except Exception:
            return ""
    try:
        ps_cmd = f"(Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\").CommandLine"
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def _tail(path: str, limit: int = 2000) -> str:
    try:
        data = Path(path).read_bytes()
        return data[-limit:].decode("utf-8", errors="replace")
    except Exception:
        return ""


class WorkerService:
    def start(self, *, backend: str, arch: str, devices: list[int], host: str, port: int) -> int:
        status = curl_status("127.0.0.1", port)
        if is_ready(status):
            print(f"[ar_cli] daemon already ready on :{port}; nothing to do")
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return 0

        timing = WorkerConfig.load(None).timing
        log_file = worker_log_path(port)
        env = os.environ.copy()
        env["WORKER_BACKEND"] = backend
        env["WORKER_ARCH"] = arch
        env["WORKER_DEVICES"] = ",".join(str(d) for d in devices)
        env["WORKER_HOST"] = host
        env["WORKER_PORT"] = str(port)
        env["AKG_WORKER_LOG_FILE"] = log_file
        env["AR_WORKER_LOG_FILE"] = log_file
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")

        log = open(log_file, "ab", buffering=0)
        kwargs = {
            "cwd": str(SCRIPTS_DIR),
            "env": env,
            "stdout": log,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "posix":
            kwargs["preexec_fn"] = os.setsid
        proc = subprocess.Popen([sys.executable, "-m", "worker.server"], **kwargs)

        quiet = os.environ.get("AR_CLI_QUIET") == "1"
        if not quiet:
            print(f"[ar_cli] starting worker backend={backend} arch={arch} devices={env['WORKER_DEVICES']} port={port}")
            print(f"[ar_cli] log: {log_file}")

        deadline = time.time() + _env_float("AR_WORKER_READY_TIMEOUT", timing.ready_timeout)
        tick = _env_float("AR_WORKER_READY_POLL_INTERVAL", timing.ready_poll_interval)
        probe_timeout = _env_float("AR_WORKER_READY_PROBE_TIMEOUT", timing.ready_probe_timeout)
        last = time.time()
        while time.time() < deadline:
            if proc.poll() is not None:
                print(f"[ar_cli] worker exited during startup rc={proc.returncode}\n{_tail(log_file)}", file=sys.stderr)
                return 1
            status = curl_status("127.0.0.1", port, timeout=probe_timeout)
            if is_ready(status):
                if quiet:
                    print(f"[ar_cli] remote worker ready pid={proc.pid} log={log_file}")
                else:
                    print(f"[ar_cli] worker ready pid={proc.pid}")
                return 0
            now = time.time()
            if not quiet and now - last >= tick:
                elapsed = int(now - (deadline - _env_float("AR_WORKER_READY_TIMEOUT", timing.ready_timeout)))
                print(f"[ar_cli] waiting for /status ready ({elapsed}s)")
                last = now
            time.sleep(1)

        print(f"[ar_cli] worker did not become ready in time\n{_tail(log_file)}", file=sys.stderr)
        return 1

    def stop(self, *, port: int) -> int:
        pid = _find_pid_on_port(port)
        if pid is None:
            print(f"[ar_cli] no listener on :{port}")
            return 0
        cmdline = _cmdline(pid)
        if "worker.server" not in cmdline and "worker/server.py" not in cmdline:
            print(f"[ar_cli] pid {pid} on :{port} does not look like worker.server", file=sys.stderr)
            return 1
        try:
            if os.name == "posix":
                os.kill(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception as exc:
            print(f"[ar_cli] cannot stop pid {pid}: {exc}", file=sys.stderr)
            return 1
        print(f"[ar_cli] stopped worker pid={pid} on :{port}")
        return 0
