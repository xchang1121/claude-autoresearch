"""Local ssh tunnel and port-owner helpers for worker remote dispatch."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

STATE_DIR = Path(os.environ.get("AR_STATE_DIR", str(Path.home() / ".autoresearch_state")))


def _tunnel_pid_path(port: int) -> Path:
    return STATE_DIR / "tunnels" / f"{port}.pid"


def tunnel_start(host: str, port: int) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "tunnels").mkdir(exist_ok=True)
    tunnel_stop_silent(port, host)

    cmd = [
        "ssh",
        "-f",
        "-N",
        "-T",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=10",
        "-L",
        f"{port}:127.0.0.1:{port}",
        host,
    ]
    stdio = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.name == "posix":
        try:
            rc = subprocess.call(cmd, **stdio)
        except Exception as exc:
            print(f"[ar_cli] ssh tunnel launch failed: {exc}", file=sys.stderr)
            return 0
        if rc != 0:
            print(f"[ar_cli] ssh tunnel exited rc={rc}; checking port state anyway", file=sys.stderr)
    else:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        try:
            subprocess.Popen(cmd, creationflags=flags, **stdio)
        except Exception as exc:
            print(f"[ar_cli] ssh tunnel spawn failed: {exc}", file=sys.stderr)
            return 0
        for _ in range(10):
            time.sleep(0.5)
            if find_tunnel_pid(port, host):
                break

    pid = find_tunnel_pid(port, host)
    if pid:
        _tunnel_pid_path(port).write_text(str(pid), encoding="ascii")
        return pid
    return 0


def find_tunnel_pid(port: int, host: str) -> int:
    if os.name == "posix":
        try:
            out = subprocess.run(
                ["pgrep", "-f", f"ssh.*-L {port}:.*{host}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in out.stdout.splitlines():
                value = line.strip()
                if value.isdigit():
                    return int(value)
        except Exception:
            return 0
        return 0

    try:
        ps_cmd = (
            f"Get-CimInstance Win32_Process -Filter \"Name='ssh.exe'\" | "
            f"Where-Object {{ $_.CommandLine -like '*-L {port}:*{host}*' }} | "
            f"Select-Object -ExpandProperty ProcessId"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in out.stdout.splitlines():
            value = line.strip()
            if value.isdigit():
                return int(value)
    except Exception:
        return 0
    return 0


def tunnel_stop_silent(port: int, host: str = "") -> None:
    pid = 0
    pid_path = _tunnel_pid_path(port)
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text(encoding="ascii").strip())
        except Exception:
            pid = 0
    if not pid and host:
        pid = find_tunnel_pid(port, host)
    if pid:
        try:
            if os.name == "posix":
                os.kill(pid, signal.SIGTERM)
            else:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    subprocess.call(["taskkill", "/PID", str(pid), "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass


def who_holds_port(port: int) -> Optional[dict]:
    if os.name == "posix":
        try:
            out = subprocess.run(
                ["lsof", "-iTCP:" + str(port), "-sTCP:LISTEN", "-Pn"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in out.stdout.splitlines()[1:]:
                parts = line.split(None, 8)
                if len(parts) >= 2 and parts[1].isdigit():
                    return {"pid": int(parts[1]), "cmdline": line.strip()}
        except Exception:
            return None
        return None

    try:
        ps_cmd = (
            f"$c = Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; "
            "$p = if ($c) { Get-CimInstance Win32_Process -Filter \"ProcessId=$($c.OwningProcess)\" }; "
            "if ($p) { Write-Output ($p.ProcessId.ToString() + ' ' + $p.CommandLine) }"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
        line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        if line:
            pid, _, cmd = line.partition(" ")
            if pid.isdigit():
                return {"pid": int(pid), "cmdline": cmd}
    except Exception:
        return None
    return None


def kill_pid_hint(pid: int) -> str:
    if os.name == "posix":
        return f"kill {pid}"
    return f"taskkill /PID {pid} /F"
