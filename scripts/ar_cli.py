#!/usr/bin/env python3
"""ar_cli — AutoResearch CLI entry.

Currently a single subcommand: `worker` (start / stop / status of the
local HTTP worker daemon). The `--remote-host <alias>` flag delegates
the same command to a remote machine via SSH and (for --start) sets up
a local ssh -L tunnel so subsequent local-port commands transparently
hit the remote daemon. Remote host config lives in
`autoresearch/config.yaml:remote_worker.hosts`.

Examples:

    # local daemon control (arch auto-derived from --backend/--devices)
    ar_cli worker --start --backend ascend --devices 6 --port 9111
    ar_cli worker --status --port 9111
    ar_cli worker --stop --port 9111

    # remote — the host alias is an entry in config.yaml. arch is
    # auto-derived on the remote side via the backend-specific probe (pass --arch
    # explicitly to override).
    ar_cli worker --remote-host my-npu --start --backend ascend \\
        --devices 6 --port 9111
    ar_cli worker --remote-host my-npu --status --port 9111
    ar_cli worker --remote-host my-npu --stop --port 9111
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
# Pin stdout/stderr to UTF-8 (user-direct entry; can't rely on parent env).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

# Run as `python scripts/ar_cli.py`, so scripts/ is on sys.path[0].
from utils.settings import (  # noqa: E402
    worker_port, worker_ready_timeout, worker_ready_poll_interval,
    worker_ready_probe_timeout, worker_status_timeout,
)

SCRIPT_DIR = Path(__file__).resolve().parent
# Autoresearch project root: scripts/ → autoresearch/
AR_ROOT = SCRIPT_DIR.parent

STATE_DIR = Path(os.environ.get(
    "AR_STATE_DIR", str(Path.home() / ".autoresearch_state")))

_LOGO_PRINTED = False
_LOGO = r"""
    _         _        ____                               _
   / \  _   _| |_ ___ |  _ \ ___  ___  ___  __ _ _ __ ___| |__
  / _ \| | | | __/ _ \| |_) / _ \/ __|/ _ \/ _` | '__/ __| '_ \
 / ___ \ |_| | || (_) |  _ <  __/\__ \  __/ (_| | | | (__| | | |
/_/   \_\__,_|\__\___/|_| \_\___||___/\___|\__,_|_|  \___|_| |_|
"""


def _color(text: str, code: str) -> str:
    if (not sys.stdout.isatty()
            or os.environ.get("NO_COLOR")
            or os.environ.get("AR_CLI_PLAIN") == "1"):
        return text
    return f"\033[{code}m{text}\033[0m"


def _print_logo_once() -> None:
    """Print the worker startup banner once for user-invoked starts."""
    global _LOGO_PRINTED
    if _LOGO_PRINTED or os.environ.get("AR_CLI_QUIET") == "1":
        return
    _LOGO_PRINTED = True
    print(_color(_LOGO.rstrip("\n"), "36"))
    print(_color("AutoResearch Worker", "1;37"))


@dataclass(frozen=True)
class Finding:
    severity: str  # ok / info / warn / fatal
    check: str
    result: str
    suggest: str = ""


def _has_fatal(findings: list[Finding]) -> bool:
    return any(f.severity == "fatal" for f in findings)


def _render_findings(findings: list[Finding], *,
                     title: str = "Diagnostics",
                     log_tail: str = "",
                     stream=None) -> None:
    stream = stream or sys.stdout
    print(f"\n{title}", file=stream)
    print("-" * max(12, len(title)), file=stream)
    sym = {"ok": "OK", "info": "INFO", "warn": "WARN", "fatal": "FAIL"}
    for f in findings:
        tag = sym.get(f.severity, f.severity.upper())
        line = f"[{tag:<4}] {f.check:<16} {f.result}"
        if f.suggest:
            line += f"  -> {f.suggest}"
        print(line, file=stream)
    if log_tail and log_tail.strip() and not log_tail.strip().startswith("(no log"):
        print("\nRemote daemon log tail:", file=stream)
        print(log_tail, file=stream)


def _load_config_yaml() -> tuple[dict, Optional[str]]:
    cfg_path = AR_ROOT / "config.yaml"
    if not cfg_path.is_file():
        return {}, None
    try:
        import yaml  # type: ignore
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}, str(cfg_path)
    except Exception as e:
        print(f"[ar_cli] failed to read {cfg_path}: {e}", file=sys.stderr)
        return {}, str(cfg_path)


def _config_default(name: str, default=None):
    data, _ = _load_config_yaml()
    return (data.get("defaults") or {}).get(name, default)


def _guess_remote_alias() -> Optional[str]:
    """If exactly one entry exists under remote_worker.hosts, return its
    alias. Used by _dispatch_worker to silently route Windows callers
    that omit --remote-host. Returns None when the config has zero or
    multiple hosts (ambiguous)."""
    data, _ = _load_config_yaml()
    hosts = ((data.get("remote_worker") or {}).get("hosts") or {})
    if len(hosts) == 1:
        return next(iter(hosts))
    return None


# ---------------------------------------------------------------------------
# Tunnel pid file helpers (used by --remote-host in Phase 2C)
# ---------------------------------------------------------------------------

def _tunnel_pid_path(port: int) -> Path:
    return STATE_DIR / "tunnels" / f"{port}.pid"


# ---------------------------------------------------------------------------
# Worker daemon — local
# ---------------------------------------------------------------------------

def _worker_log_path(port: int) -> str:
    return f"/tmp/ar_worker_{port}.log"


def _curl_status(host: str, port: int,
                 timeout: float = None) -> Optional[dict]:
    if timeout is None:
        timeout = worker_status_timeout()
    url = f"http://{host}:{port}/api/v1/status"
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (URLError, socket.timeout, ConnectionError, Exception):
        return None


def _curl_health(host: str, port: int,
                 timeout: float = 6.0) -> Optional[dict]:
    """GET /api/v1/health —— /status 只验 HTTP 在线，/health 真走一遍
    device acquire/release，能抓出 "status 回 200 但 /run 永远卡住" 类的
    handler deadlock。timeout 比 worker 侧的 5s 探活多 1s 给网络往返用。
    旧版 daemon 没有 /health 端点 → urlopen 抛 HTTPError 404 → 返回 None，
    调用方退化为只看 /status。"""
    url = f"http://{host}:{port}/api/v1/health"
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (URLError, socket.timeout, ConnectionError, Exception):
        return None


def _find_pid_on_port(port: int) -> Optional[int]:
    """POSIX-only: returns the PID listening on TCP `port`, or None.

    Tries `ss -ltnp` first (modern), falls back to `lsof -iTCP:port`.
    Both produce process info; we parse the first numeric pid we see.
    """
    if os.name != "posix":
        return None

    # `ss -ltnp` output line example (worker binds loopback):
    #   LISTEN 0  2048  127.0.0.1:9111  127.0.0.1:*  users:(("python",pid=12345,fd=7))
    try:
        out = subprocess.run(
            ["ss", "-ltnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=5,
        )
        for ln in out.stdout.splitlines():
            if f":{port} " in ln or ln.rstrip().endswith(f":{port}"):
                # extract pid=NNN
                m = ln.split("pid=")
                if len(m) > 1:
                    pid_part = m[1].split(",")[0].split(")")[0]
                    return int(pid_part)
    except Exception:
        pass

    try:
        out = subprocess.run(
            ["lsof", "-iTCP:" + str(port), "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5,
        )
        for ln in out.stdout.splitlines():
            ln = ln.strip()
            if ln.isdigit():
                return int(ln)
    except Exception:
        pass

    return None


def _cmdline_contains_worker(pid: int) -> bool:
    """Safety check before SIGTERM — only kill if /proc/<pid>/cmdline
    looks like our worker. POSIX-only."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\x00", b" ").decode(errors="replace")
        return "worker.server" in cmd or "worker/server.py" in cmd
    except Exception:
        return False


def cmd_worker_start(args) -> int:
    # arch is optional on the CLI; auto-derive from the first --devices
    # entry using the backend-specific probe. Caller can still pass --arch
    # explicitly to override.
    if not args.arch:
        from utils.hw_detect import derive_arch, probe_hint
        try:
            first_dev = int(args.devices.split(",")[0].strip())
        except (ValueError, AttributeError):
            print(f"[ar_cli] --start: cannot parse first device id from "
                  f"--devices={args.devices!r}", file=sys.stderr)
            return 2
        # Route the probe by backend (ascend→npu-smi, cuda→nvidia-smi,
        # cpu→platform.machine). Previously hard-coded "ascend" silently
        # ran npu-smi for --backend cuda too, then errored with a
        # misleading "npu-smi missing" hint.
        derived = derive_arch(first_dev, args.backend)
        if not derived:
            print(f"[ar_cli] --start: arch auto-derive failed for "
                  f"backend={args.backend!r} device={first_dev} "
                  f"({probe_hint(args.backend)}). Pass --arch explicitly.",
                  file=sys.stderr)
            return 2
        args.arch = derived

    # Idempotency: skip spawn if daemon already alive.
    if _curl_status("127.0.0.1", args.port):
        print(f"[ar_cli] daemon already alive on :{args.port}; nothing to do")
        return 0

    env = os.environ.copy()
    env["WORKER_BACKEND"] = args.backend
    env["WORKER_ARCH"] = args.arch
    env["WORKER_DEVICES"] = args.devices
    env["WORKER_HOST"] = "0.0.0.0"
    env["WORKER_PORT"] = str(args.port)
    env["PYTHONIOENCODING"] = "utf-8"  # propagates UTF-8 to worker.server + its eval subprocs

    cmd = [sys.executable, "-m", "worker.server"]

    log_path = _worker_log_path(args.port)
    log_fh = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(SCRIPT_DIR),
        env=env,
        stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    # Poll /status until ready or timeout — gives a clear failure
    # message instead of "fork succeeded, daemon crashed silently".
    ready_timeout = worker_ready_timeout()
    deadline = time.time() + ready_timeout
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            break  # process died during boot
        if _curl_status("127.0.0.1", args.port,
                        timeout=worker_ready_probe_timeout()):
            ready = True
            break
        time.sleep(worker_ready_poll_interval())

    if not ready:
        rc = proc.poll()
        tail = ""
        try:
            with open(log_path, "rb") as f:
                tail = f.read()[-1500:].decode(errors="replace")
        except Exception:
            pass
        print(f"[ar_cli] worker on :{args.port} did not become ready "
              f"within {ready_timeout:g}s (process rc={rc}). "
              f"Log tail:\n{tail}",
              file=sys.stderr)
        return 1

    print("=" * 56)
    print("AutoResearch Worker (daemon)")
    print("-" * 56)
    print(f"  Backend : {args.backend}")
    print(f"  Arch    : {args.arch}")
    print(f"  Devices : {args.devices}")
    print(f"  Port    : {args.port}")
    print(f"  PID     : {proc.pid}")
    print(f"  Log     : {log_path}")
    print("=" * 56)
    return 0


def cmd_worker_stop(args) -> int:
    pid = _find_pid_on_port(args.port)
    if pid is None:
        # Already stopped → idempotent success. Matches systemctl stop /
        # docker stop convention so `--stop; --stop` doesn't pretend the
        # second call failed.
        print(f"[ar_cli] no listener on :{args.port}", file=sys.stderr)
        return 0

    if not args.force and not _cmdline_contains_worker(pid):
        print(f"[ar_cli] pid {pid} on :{args.port} does not look like a "
              f"worker (cmdline missing 'worker.server'). Pass --force "
              f"to kill anyway.", file=sys.stderr)
        return 1

    try:
        # SIGTERM, wait briefly, SIGKILL if still alive.
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.25)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                print(f"[ar_cli] stopped worker pid={pid} on :{args.port}")
                return 0
        os.kill(pid, signal.SIGKILL)
        print(f"[ar_cli] SIGKILLed unresponsive worker pid={pid} on :{args.port}",
              file=sys.stderr)
        return 0
    except ProcessLookupError:
        print(f"[ar_cli] pid {pid} already gone")
        return 0
    except PermissionError as e:
        print(f"[ar_cli] cannot signal pid {pid}: {e}", file=sys.stderr)
        return 1


def cmd_worker_status(args) -> int:
    """status 同时探 /status + /health。/status 只验 HTTP server 在线；
    /health 走一次真实 device acquire/release 抓 handler deadlock。旧版 daemon
    /health 端点不存在时返 None，退化为只看 /status。纯查询，无副作用。"""
    host = "127.0.0.1"
    st = _curl_status(host, args.port)
    if st is None:
        print(f"Worker on {host}:{args.port} unreachable.", file=sys.stderr)
        _run_local_diagnostics(
            args.port,
            backend=getattr(args, "backend", None),
            dsl=getattr(args, "dsl", None),
            stream=sys.stderr,
        )
        return 1
    health = _curl_health(host, args.port)
    out = dict(st)
    if health is not None:
        out["health"] = {"healthy": bool(health.get("healthy")),
                         "probed_device": health.get("probed_device"),
                         "error": health.get("error")}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if health is not None and not health.get("healthy"):
        print(
            f"\n[ar_cli] /status OK 但 /health 报 degraded —— "
            f"daemon handler 可能处于 deadlock 状态。错误：{health.get('error')!r}",
            file=sys.stderr,
        )
        return 1
    return 0


def _first_device_id(devices: Optional[str]) -> Optional[int]:
    if not devices:
        return None
    try:
        first = str(devices).split(",", 1)[0].strip()
        return int(first) if first else None
    except (TypeError, ValueError):
        return None


_CONDA_HOOK_BASH = r"""
if command -v conda >/dev/null 2>&1; then
  __ar_conda_base="$(conda info --base 2>/dev/null || true)"
  if [ -n "$__ar_conda_base" ] && [ -f "$__ar_conda_base/etc/profile.d/conda.sh" ]; then
    . "$__ar_conda_base/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
  else
    eval "$(conda shell.bash hook 2>/dev/null)" >/dev/null 2>&1 || true
  fi
  unset __ar_conda_base
fi
""".strip()


def _source_env_script_bash(env_script: Optional[str]) -> str:
    parts = [_CONDA_HOOK_BASH]
    if env_script:
        parts.append(f"source {shlex.quote(env_script)}")
    return "\n".join(parts)


def _source_env_var_bash(var_name: str) -> str:
    return "\n".join([
        _CONDA_HOOK_BASH,
        f'if [ -n "${var_name}" ] && [ -f "${var_name}" ]; then',
        f'  source "${var_name}" >/dev/null 2>&1',
        "fi",
    ])


_REMOTE_PROBE_BASH = r"""
env_script={env_script}
backend={backend}
probe_device={probe_device}
port={port}
log_file={log_file}
echo "ENV_PATH:$env_script"
echo "PROBE_BACKEND:$backend"
echo "PROBE_DEVICE:$probe_device"
if [ -n "$env_script" ]; then
  [ -f "$env_script" ] && echo "ENV_OK:yes" || echo "ENV_OK:no"
else
  echo "ENV_OK:"
fi
{env_setup}
TORCH_NPU_OUT=$(python -c 'import torch_npu' 2>&1); TORCH_NPU_RC=$?
if [ $TORCH_NPU_RC -eq 0 ]; then
  echo "TORCH_NPU:ok"
else
  echo "TORCH_NPU:$(echo "$TORCH_NPU_OUT" | tail -1)"
fi
TRITON_OUT=$(python -c 'import triton' 2>&1); TRITON_RC=$?
if [ $TRITON_RC -eq 0 ]; then
  echo "TRITON:ok"
else
  echo "TRITON:$(echo "$TRITON_OUT" | tail -1)"
fi
if [ "$backend" = "cuda" ]; then
  echo "NPU_SMI:not_required"
  echo "NVIDIA_SMI:$(command -v nvidia-smi >/dev/null 2>&1 && echo ok || echo missing)"
  if [ -n "$probe_device" ]; then
    echo "ARCH:$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$probe_device" 2>/dev/null | head -1)"
  else
    echo "ARCH:$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)"
  fi
  echo "DEVICES:$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)"
elif [ "$backend" = "cpu" ]; then
  echo "NPU_SMI:not_required"
  echo "NVIDIA_SMI:not_required"
  echo "ARCH:$(python -c 'import platform; print((platform.machine() or "").lower())' 2>/dev/null)"
  echo "DEVICES:1"
else
  echo "NPU_SMI:$(command -v npu-smi >/dev/null 2>&1 && echo ok || echo missing)"
  echo "NVIDIA_SMI:not_required"
  echo "ARCH:$(npu-smi info 2>/dev/null | awk -v did="$probe_device" '/^\| +[0-9]+ +[0-9A-Z]/{{if (did == "" || $2 == did) {{print $3; exit}}}}')"
  echo "DEVICES:$(npu-smi info 2>/dev/null | grep -cE '^\| +[0-9]+ +[0-9A-Z]')"
fi
if command -v lsof >/dev/null 2>&1; then
  echo "PORT_PID:$(lsof -ti :$port -sTCP:LISTEN 2>/dev/null | head -1)"
elif command -v ss >/dev/null 2>&1; then
  echo "PORT_PID:$(ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)"
else
  echo "PORT_PID:"
fi
echo "DISK_FREE_MB:$(df -kP /tmp / 2>/dev/null | awk 'NR>1 {{print int($4/1024)}}' | sort -n | head -1)"
echo "LOG_TAIL_BEGIN"
[ -f "$log_file" ] && tail -20 "$log_file" || echo "(no log: $log_file)"
"""


def _probe_remote(ssh_alias: str, env_script: Optional[str],
                  port: int, *, backend: Optional[str] = None,
                  devices: Optional[str] = None) -> dict:
    probe_device = _first_device_id(devices)
    probe = _REMOTE_PROBE_BASH.format(
        env_script=shlex.quote(env_script or ""),
        backend=shlex.quote((backend or "ascend").lower()),
        probe_device=shlex.quote("" if probe_device is None else str(probe_device)),
        port=int(port),
        log_file=shlex.quote(_worker_log_path(port)),
        env_setup=_source_env_var_bash("env_script"),
    )
    try:
        out = subprocess.run(
            [
                "ssh",
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                ssh_alias,
                f"bash -lc {shlex.quote(probe)}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=35,
        )
    except subprocess.TimeoutExpired:
        return {"_SSH_ERROR": "ssh probe timed out after 35s"}
    except Exception as e:
        return {"_SSH_ERROR": str(e)[:200]}

    if out.returncode != 0:
        err = (out.stderr or "").strip() or f"ssh exit rc={out.returncode}"
        return {"_SSH_ERROR": err[:200]}

    facts: dict = {}
    log_lines: list[str] = []
    in_log = False
    for line in out.stdout.splitlines():
        if in_log:
            log_lines.append(line)
        elif line == "LOG_TAIL_BEGIN":
            in_log = True
        elif ":" in line:
            key, value = line.split(":", 1)
            facts[key] = value.strip()
    facts["LOG_TAIL"] = "\n".join(log_lines)
    return facts


def _ssh_suggestion(err: str) -> str:
    low = err.lower()
    if "could not resolve hostname" in low or "name or service not known" in low:
        return "check ~/.ssh/config alias"
    if "timed out" in low or "no route to host" in low:
        return "check VPN/network/routing"
    if "permission denied" in low or "publickey" in low:
        return "check ssh key / authorized_keys"
    if "host key verification failed" in low:
        return "run ssh-keygen -R <host> if host key changed"
    return "try `ssh <alias>` manually for the raw error"


def _classify_probe(facts: dict, port: int, *,
                    backend: Optional[str],
                    dsl: Optional[str],
                    for_start: bool) -> list[Finding]:
    ssh_err = facts.get("_SSH_ERROR")
    if ssh_err:
        return [Finding("fatal", "ssh", ssh_err, _ssh_suggestion(ssh_err))]

    backend_n = (backend or "").strip().lower()
    ascendish = backend_n in ("", "ascend")
    needs_triton = (dsl or "").strip().lower().startswith("triton")
    findings: list[Finding] = []

    env_path = facts.get("ENV_PATH") or ""
    env_ok = facts.get("ENV_OK") or ""
    if not env_path:
        findings.append(Finding(
            "info", "env_script", "not configured",
            "ok only if the remote login shell already initializes CANN",
        ))
    elif env_ok == "yes":
        findings.append(Finding("ok", "env_script", env_path))
    else:
        findings.append(Finding(
            "fatal", "env_script", f"{env_path} missing",
            "fix config.yaml remote_worker.hosts.<alias>.env_script",
        ))

    torch_npu = facts.get("TORCH_NPU") or ""
    if torch_npu == "ok":
        findings.append(Finding("ok", "torch_npu", "importable"))
    elif ascendish:
        findings.append(Finding(
            "fatal", "torch_npu", torch_npu[:120] or "import failed",
            "source CANN env or install torch_npu",
        ))
    else:
        findings.append(Finding("info", "torch_npu", "not required"))

    triton = facts.get("TRITON") or ""
    if triton == "ok":
        findings.append(Finding("ok", "triton", "importable"))
    else:
        findings.append(Finding(
            "fatal" if needs_triton else "warn",
            "triton",
            triton[:100] or "import failed",
            "required for triton_* DSLs",
        ))

    if facts.get("NPU_SMI") == "ok":
        findings.append(Finding("ok", "npu-smi", "in PATH"))
    elif ascendish:
        findings.append(Finding(
            "fatal", "npu-smi", "missing",
            "source CANN set_env.sh in env_script",
        ))
    else:
        findings.append(Finding("info", "npu-smi", "not required"))

    if backend_n == "cuda":
        if facts.get("NVIDIA_SMI") == "ok":
            findings.append(Finding("ok", "nvidia-smi", "in PATH"))
        else:
            findings.append(Finding(
                "fatal", "nvidia-smi", "missing",
                "check CUDA driver and PATH on the remote",
            ))

    arch = (facts.get("ARCH") or "").strip()
    if arch:
        if backend_n == "ascend":
            findings.append(Finding("ok", "npu arch", f"ascend{arch.lower()}"))
        elif backend_n == "cuda":
            findings.append(Finding("ok", "cuda gpu", arch))
        elif backend_n == "cpu":
            findings.append(Finding("ok", "cpu arch", arch))
        else:
            findings.append(Finding("ok", "arch", arch))
    elif ascendish:
        findings.append(Finding(
            "warn", "npu arch", "not detected",
            "pass --arch explicitly if auto-detect fails",
        ))
    elif backend_n == "cuda":
        findings.append(Finding(
            "warn", "cuda gpu", "not detected",
            "pass --arch explicitly if auto-detect fails",
        ))
    elif backend_n == "cpu":
        findings.append(Finding(
            "warn", "cpu arch", "not detected",
            "platform.machine() returned empty",
        ))

    try:
        ndev = int(facts.get("DEVICES") or "0")
    except ValueError:
        ndev = 0
    if ndev > 0:
        if backend_n == "cuda":
            label = "cuda devices"
        elif backend_n == "cpu":
            label = "cpu slots"
        else:
            label = "npu devices"
        findings.append(Finding("ok", label, f"{ndev} visible"))
    elif ascendish:
        findings.append(Finding(
            "fatal", "npu devices", "0 visible",
            "check driver and `npu-smi info` on the remote",
        ))
    elif backend_n == "cuda":
        findings.append(Finding(
            "fatal", "cuda devices", "0 visible",
            "check driver and `nvidia-smi` on the remote",
        ))

    try:
        free_mb = int(facts.get("DISK_FREE_MB") or "0")
    except ValueError:
        free_mb = 0
    if free_mb >= 500:
        findings.append(Finding("ok", "disk free", f"{free_mb} MB"))
    elif free_mb > 0:
        findings.append(Finding(
            "fatal", "disk free", f"only {free_mb} MB",
            "clear /tmp or remote logs before starting worker",
        ))

    port_pid = (facts.get("PORT_PID") or "").strip()
    if port_pid:
        findings.append(Finding(
            "fatal" if for_start else "warn",
            f"remote :{port}",
            f"held by PID {port_pid}",
            "stop stale daemon or choose another --port",
        ))
    else:
        findings.append(Finding("ok", f"remote :{port}", "free"))

    return findings


def _run_remote_diagnostics(alias: str, host_cfg: dict, port: int, *,
                            backend: Optional[str],
                            devices: Optional[str],
                            dsl: Optional[str],
                            for_start: bool,
                            stream=None) -> bool:
    ssh_alias = host_cfg.get("ssh_alias") or alias
    facts = _probe_remote(
        ssh_alias, host_cfg.get("env_script"), port,
        backend=backend, devices=devices,
    )
    findings = _classify_probe(
        facts, port, backend=backend, dsl=dsl, for_start=for_start)
    _render_findings(
        findings,
        title=f"Remote diagnostics: {alias}",
        log_tail=facts.get("LOG_TAIL", ""),
        stream=stream or sys.stdout,
    )
    return not _has_fatal(findings)


def _run_local_diagnostics(port: int, *,
                           backend: Optional[str],
                           dsl: Optional[str],
                           stream=None) -> bool:
    backend_n = (backend or _config_default("backend", "") or "").lower()
    dsl_n = (dsl or _config_default("dsl", "") or "")
    findings: list[Finding] = []
    _, cfg_path = _load_config_yaml()
    findings.append(Finding(
        "ok" if cfg_path else "warn",
        "config.yaml",
        cfg_path or "not found",
        "" if cfg_path else "run from repo root or pass explicit flags",
    ))
    findings.append(Finding(
        "ok", "python",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    ))
    status = _curl_status("127.0.0.1", port)
    findings.append(Finding(
        "ok" if status else "info",
        f"local :{port}",
        "worker reachable" if status else "no worker responding",
    ))

    if backend_n == "ascend":
        try:
            smi = subprocess.run(
                ["npu-smi", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
            )
            findings.append(Finding(
                "ok" if smi.returncode == 0 else "fatal",
                "npu-smi",
                "ok" if smi.returncode == 0 else (smi.stderr or "failed")[:100],
                "" if smi.returncode == 0 else "source CANN set_env.sh",
            ))
        except Exception as e:
            findings.append(Finding(
                "fatal", "npu-smi", str(e)[:100],
                "source CANN set_env.sh",
            ))
        torch = subprocess.run(
            [sys.executable, "-c", "import torch_npu"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        findings.append(Finding(
            "ok" if torch.returncode == 0 else "fatal",
            "torch_npu",
            "importable" if torch.returncode == 0 else (torch.stderr or "failed")[-120:],
        ))

    if str(dsl_n).startswith("triton"):
        tri = subprocess.run(
            [sys.executable, "-c", "import triton"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        findings.append(Finding(
            "ok" if tri.returncode == 0 else "fatal",
            "triton",
            "importable" if tri.returncode == 0 else (tri.stderr or "failed")[-120:],
        ))

    _render_findings(
        findings,
        title="Local diagnostics",
        stream=stream or sys.stdout,
    )
    return not _has_fatal(findings)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ar_cli",
        description="AutoResearch CLI.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser(
        "worker",
        help="Start / stop / check the AutoResearch worker daemon.",
        description=(
            "Start / stop / check the AutoResearch worker on this "
            "machine (or, with --remote-host, on a config-defined "
            "remote machine via SSH)."
        ),
    )
    action = w.add_mutually_exclusive_group(required=True)
    action.add_argument("--start", action="store_true",
                        help="Start the worker. Idempotent: skips spawn "
                             "if daemon is already alive; rebuilds the "
                             "ssh -L tunnel on demand for --remote-host.")
    action.add_argument("--stop", action="store_true",
                        help="Stop the daemon listening on --port.")
    action.add_argument("--status", action="store_true",
                        help="Probe /api/v1/status + /health. If the "
                             "worker is unreachable, print local or remote "
                             "diagnostics without spawning a daemon.")

    w.add_argument("--backend", choices=["ascend", "cuda", "cpu"],
                   help="Hardware backend (required for --start).")
    w.add_argument("--arch",
                   help="Arch string, e.g. ascend910b3. Optional for "
                        "--start defaults to auto-derive via the "
                        "backend-specific probe on the first --devices entry. Pass "
                        "explicitly to override.")
    w.add_argument("--devices",
                   help="Comma-separated device IDs, e.g. '2,5' "
                        "(required for --start).")
    w.add_argument("--dsl",
                   help="Target DSL for status diagnostics. triton_* makes "
                        "missing triton fatal; other DSLs warn only.")
    w.add_argument("--port", type=int, default=worker_port(),
                   help=f"TCP port (default: {worker_port()}, "
                        f"from config.yaml worker.port).")
    w.add_argument("--force", action="store_true",
                   help="For --stop: skip the worker-cmdline safety check.")

    w.add_argument("--remote-host", default=None,
                   help="Run the same worker command on a remote host "
                        "(SSH alias defined in autoresearch/config.yaml:"
                        "remote_worker.hosts). When --start, also opens "
                        "a local ssh -L tunnel so 127.0.0.1:<port> "
                        "forwards to the remote daemon.")
    w.set_defaults(func=_dispatch_worker)
    return ap


def _validate_worker_args(args) -> Optional[str]:
    if args.start and not (args.backend and args.devices):
        return ("--start requires --backend and --devices "
                "(--arch is auto-derived from the backend probe when omitted).")
    return None


def _dispatch_worker(args) -> int:
    err = _validate_worker_args(args)
    if err:
        print(f"[ar_cli] {err}", file=sys.stderr)
        return 2

    if args.start:
        _print_logo_once()

    # Windows: local worker has no implementation (the daemon needs
    # os.setsid; the stop path needs ss/lsof+SIGTERM). Force every
    # `python scripts/ar_cli.py worker --...` through the remote
    # dispatcher. Auto-fill --remote-host from the only configured
    # alias; if zero or multiple, surface the config issue directly
    # instead of leaking POSIX details into the error.
    if os.name != "posix" and not args.remote_host:
        alias = _guess_remote_alias()
        if alias:
            args.remote_host = alias
        else:
            data, _ = _load_config_yaml()
            hosts = ((data.get("remote_worker") or {}).get("hosts") or {})
            if not hosts:
                print("[ar_cli] no remote_worker.hosts in config.yaml.",
                      file=sys.stderr)
            else:
                print(f"[ar_cli] multiple remote_worker.hosts configured "
                      f"({', '.join(hosts)}); pass --remote-host <alias>.",
                      file=sys.stderr)
            return 2

    if args.remote_host:
        return _dispatch_remote_worker(args)

    if args.start:
        return cmd_worker_start(args)
    if args.stop:
        return cmd_worker_stop(args)
    if args.status:
        return cmd_worker_status(args)
    return 2


# ---------------------------------------------------------------------------
# Remote worker — SSH dispatch + ssh -L tunnel
# ---------------------------------------------------------------------------

def _load_remote_host_config(alias: str) -> Optional[dict]:
    """Look up alias under autoresearch/config.yaml :: remote_worker.hosts.
    Returns None if config or alias is missing — caller surfaces the error."""
    cfg_path = AR_ROOT / "config.yaml"
    if not cfg_path.is_file():
        return None
    try:
        import yaml  # type: ignore
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[ar_cli] failed to read {cfg_path}: {e}", file=sys.stderr)
        return None
    hosts = (((data.get("remote_worker") or {}).get("hosts") or {}))
    return hosts.get(alias)


def _build_remote_ar_cli_cmd(host_cfg: dict, ar_cli_args: list[str]) -> str:
    """Compose the bash command we send through ssh: bootstrap env, cd repo,
    invoke the remote ar_cli.py with the equivalent (non-remote) args.

    All values are shlex-quoted; the resulting string is passed to ssh
    AS A SINGLE ARG so the remote shell parses it as one command.
    """
    repo_path = host_cfg["repo_path"]  # required; KeyError surfaces cleanly
    env_script = host_cfg.get("env_script")

    parts: list[str] = [_source_env_script_bash(env_script)]
    parts.append("export AR_CLI_QUIET=1")
    parts.append(f"cd {shlex.quote(repo_path)}")
    parts.append(
        "python scripts/ar_cli.py "
        + " ".join(shlex.quote(a) for a in ar_cli_args)
    )
    return "\n".join(parts)


def _strip_remote_flags(args) -> list[str]:
    """Reconstruct the non-remote ar_cli worker args for the remote side.
    Mirrors the parser flags exactly so the remote ar_cli runs the same
    code path as if the user typed it directly there.

    The remote daemon always binds 0.0.0.0 (by env in cmd_worker_start);
    the local ssh -L tunnel reaches it via the remote's 127.0.0.1 anyway.
    """
    out = ["worker"]
    if args.start:
        out.append("--start")
    elif args.stop:
        out.append("--stop")
    elif args.status:
        out.append("--status")

    if args.backend:
        out += ["--backend", args.backend]
    if args.arch:
        out += ["--arch", args.arch]
    if args.devices:
        out += ["--devices", args.devices]
    if getattr(args, "dsl", None):
        out += ["--dsl", args.dsl]
    out += ["--port", str(args.port)]
    if args.force:
        out.append("--force")
    return out


def _tunnel_start(host: str, port: int) -> int:
    """Start `ssh -f -N -T -L <port>:127.0.0.1:<port> <host>`, stash the
    forked pid. Returns the pid on success, 0 on soft failure.

    On Windows, `ssh -f` doesn't reliably daemonize when invoked via
    Python subprocess — the parent ssh.exe stays attached and
    subprocess.call() blocks forever even with all 3 std streams to
    DEVNULL. Sidestep with DETACHED_PROCESS + NEW_PROCESS_GROUP via
    Popen so ssh.exe runs independently from ar_cli's console, then
    poll for the tunnel pid via cmdline scan. On POSIX, ssh -f
    detaches correctly via setsid + we can keep subprocess.call.

    This mirrors akg_agents/python/akg_agents/cli/service/tunnel.py so
    both CLIs spawn the worker tunnel the same way.

    Does NOT pass `ExitOnForwardFailure=yes` — user `~/.ssh/config`
    may declare unrelated RemoteForward entries whose failure shouldn't
    take down the -L we need. Readiness is confirmed via a curl probe
    to /api/v1/status after the spawn."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "tunnels").mkdir(exist_ok=True)
    pid_path = _tunnel_pid_path(port)

    # If a stale tunnel exists, tear it down first so the new one binds.
    _tunnel_stop_silent(port, host)

    # Keepalive bumped from OpenSSH default 60s/3x → 30s/10x (~5min idle
    # tolerance) so long PLAN phases don't lose the tunnel between evals.
    cmd = [
        "ssh", "-f", "-N", "-T",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=10",
        "-L", f"{port}:127.0.0.1:{port}",
        host,
    ]
    kwargs = dict(
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if os.name == "posix":
        try:
            rc = subprocess.call(cmd, **kwargs)
        except Exception as e:
            print(f"[ar_cli] ssh tunnel launch failed: {e}", file=sys.stderr)
            return 0
        if rc != 0:
            print(f"[ar_cli] ssh exited rc={rc} (unrelated forward may have "
                  f"failed; checking -L {port} via status probe).",
                  file=sys.stderr)
    else:
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP
                 | getattr(subprocess, "DETACHED_PROCESS", 0x00000008))
        try:
            subprocess.Popen(cmd, creationflags=flags, **kwargs)
        except Exception as e:
            print(f"[ar_cli] ssh tunnel spawn failed: {e}", file=sys.stderr)
            return 0
        # Poll up to ~5s for ssh to authenticate + bind the local port.
        for _ in range(10):
            time.sleep(0.5)
            if _find_tunnel_pid(port, host):
                break

    pid = _find_tunnel_pid(port, host)
    if pid:
        pid_path.write_text(str(pid))
        return pid

    print(f"[ar_cli] tunnel established but pid not captured; manual "
          f"cleanup needed if --stop misses it.", file=sys.stderr)
    return 0


def _find_tunnel_pid(port: int, host: str) -> int:
    """Find the ssh process holding the `-L <port>:127.0.0.1:<port> <host>`
    tunnel by scanning ssh.exe / ssh process command lines. Cross-platform:
    pgrep on POSIX, PowerShell Win32_Process on Windows. Returns 0 when
    not found (caller treats as a soft failure)."""
    if os.name == "posix":
        try:
            out = subprocess.run(
                ["pgrep", "-f", f"ssh.*-L {port}:.*{host}"],
                capture_output=True, text=True, timeout=5,
            )
            for ln in out.stdout.splitlines():
                ln = ln.strip()
                if ln.isdigit():
                    return int(ln)
        except Exception:
            pass
        return 0

    # Windows: Get-CimInstance Win32_Process filtered by Name + cmdline.
    try:
        ps_cmd = (
            f"Get-CimInstance Win32_Process -Filter \"Name='ssh.exe'\" | "
            f"Where-Object {{ $_.CommandLine -like '*-L {port}:*{host}*' }} | "
            f"Select-Object -ExpandProperty ProcessId"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=10,
        )
        for ln in out.stdout.splitlines():
            ln = ln.strip()
            if ln.isdigit():
                return int(ln)
    except Exception:
        pass
    return 0


def _tunnel_stop_silent(port: int, host: str = "") -> None:
    """Best-effort tunnel teardown. Prefer the pid stashed at start time;
    fall back to a fresh cmdline scan (handles the case where --start
    couldn't capture the pid, e.g. first time setting up on Windows)."""
    pid_path = _tunnel_pid_path(port)
    pid = 0
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text().strip())
        except (ValueError, FileNotFoundError):
            pid = 0

    if pid == 0 and host:
        pid = _find_tunnel_pid(port, host)

    if pid:
        try:
            if os.name == "posix":
                os.kill(pid, signal.SIGTERM)
            else:
                # SIGTERM works on Windows for non-elevated procs since
                # 3.2; fall back to taskkill for stragglers.
                try:
                    os.kill(pid, signal.SIGTERM)
                except (PermissionError, OSError):
                    subprocess.call(["taskkill", "/PID", str(pid), "/F"])
        except (ProcessLookupError, ValueError, FileNotFoundError):
            pass
        except Exception as e:
            print(f"[ar_cli] tunnel stop failed: {e}", file=sys.stderr)

    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass


def _dispatch_remote_worker(args) -> int:
    host_cfg = _load_remote_host_config(args.remote_host)
    if host_cfg is None:
        print(f"[ar_cli] no remote_worker.hosts.{args.remote_host} entry "
              f"in autoresearch/config.yaml.", file=sys.stderr)
        return 2
    if "repo_path" not in host_cfg:
        print(f"[ar_cli] remote_worker.hosts.{args.remote_host} missing "
              f"`repo_path`.", file=sys.stderr)
        return 2

    ssh_alias = host_cfg.get("ssh_alias") or args.remote_host

    if args.status:
        st = _curl_status("127.0.0.1", args.port)
        if st is None:
            print(f"Worker tunnel 127.0.0.1:{args.port} unreachable "
                  f"(run `--start` to (re)build it).",
                  file=sys.stderr)
            _run_remote_diagnostics(
                args.remote_host,
                host_cfg,
                args.port,
                backend=getattr(args, "backend", None),
                devices=getattr(args, "devices", None),
                dsl=getattr(args, "dsl", None),
                for_start=False,
                stream=sys.stderr,
            )
            return 1
        print(json.dumps(st, indent=2))
        return 0

    if args.stop:
        remote_args = _strip_remote_flags(args)
        remote_cmd = _build_remote_ar_cli_cmd(host_cfg, remote_args)
        print(f"[ar_cli] remote ({ssh_alias}): {remote_cmd}", file=sys.stderr)
        rc = subprocess.call(
            ["ssh", ssh_alias, f"bash -lc {shlex.quote(remote_cmd)}"])
        _tunnel_stop_silent(args.port, ssh_alias)
        print(f"[ar_cli] tore down local tunnel for :{args.port}")
        return rc

    # args.start path — idempotent:
    #   1. probe 127.0.0.1:port via existing tunnel → alive: done
    #   2. rebuild tunnel + re-probe → alive: daemon was fine, tunnel was dead
    #   3. SSH-spawn remote daemon, then ensure tunnel is up
    st = _curl_status("127.0.0.1", args.port)
    if st is not None:
        print(f"[ar_cli] daemon at 127.0.0.1:{args.port} already alive; "
              f"nothing to do")
        print(json.dumps(st, indent=2))
        return 0

    _tunnel_stop_silent(args.port, ssh_alias)
    pid = _tunnel_start(ssh_alias, args.port)
    if pid:
        print(f"[ar_cli] ssh -L 127.0.0.1:{args.port} -> "
              f"{ssh_alias}:{args.port} (tunnel pid={pid})")
    st = _curl_status("127.0.0.1", args.port)
    if st is not None:
        print(f"[ar_cli] tunnel rebuilt; daemon was already running remotely")
        print(json.dumps(st, indent=2))
        return 0

    print(f"[ar_cli] running remote preflight for {args.remote_host}...",
          file=sys.stderr)
    ok = _run_remote_diagnostics(
        args.remote_host,
        host_cfg,
        args.port,
        backend=args.backend,
        devices=args.devices,
        dsl=getattr(args, "dsl", None),
        for_start=True,
        stream=sys.stderr,
    )
    if not ok:
        print("[ar_cli] remote preflight failed; worker was not spawned.",
              file=sys.stderr)
        return 1

    # Need to actually spawn the remote daemon.
    remote_args = _strip_remote_flags(args)
    remote_cmd = _build_remote_ar_cli_cmd(host_cfg, remote_args)
    print(f"[ar_cli] remote ({ssh_alias}): {remote_cmd}", file=sys.stderr)
    rc = subprocess.call(
        ["ssh", ssh_alias, f"bash -lc {shlex.quote(remote_cmd)}"])
    if rc != 0:
        print(f"[ar_cli] remote ar_cli exited rc={rc}", file=sys.stderr)
        return rc
    st = _curl_status("127.0.0.1", args.port)
    if st is None:
        print(f"[ar_cli] tunneled status probe failed after spawn; remote "
              f"daemon may not be ready or tunnel didn't bind.",
              file=sys.stderr)
        return 1
    print(json.dumps(st, indent=2))
    return 0


def main() -> int:
    ap = _build_parser()
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
