#!/usr/bin/env python3
"""AutoResearch CLI — single entry for worker / verify / env subcommands.

Canonical invocations (from project root):

    # worker
    python .autoresearch/scripts/ar_cli.py worker --start \
        --backend ascend --arch ascend910b3 --devices 2,5 \
        --host 127.0.0.1 --port 9111 --bg

    # JSON-in / JSON-out verify + profile (sentinel: AR_VERIFY_RESULT:)
    python .autoresearch/scripts/ar_cli.py verify \
        --task-config @task.yaml   \
        --impl       @kernel.py    \
        --reference  @ref.py       \
        [--task-dir <dir>]         \
        [--device-id N | --worker-url URL] \
        [--mode verify+profile|verify-only|profile-only]

    # environment introspection (sentinel: AR_ENV_RESULT:)
    python .autoresearch/scripts/ar_cli.py env detect [--worker-url URL]
    python .autoresearch/scripts/ar_cli.py env check  --framework torch --backend ascend --dsl triton_ascend
    python .autoresearch/scripts/ar_cli.py env list-dsls

`verify` materialises impl/ref into a tempdir so internal callers
(pipeline.py / baseline.py / batch/verify.py) and external callers
(any subprocess driver) share one stable JSON interface. `--task-dir`
is the escape hatch when the original task layout has support .py
files alongside kernel.py (editable_files) or `.ar_state/progress.json`
for sticky-baseline override — they get copied into the tempdir.

Platform support:
  - `worker --start` (foreground) and `worker --status` work on POSIX
    and Windows alike.
  - `worker --start --bg` (daemon mode) detaches on both platforms
    (start_new_session on POSIX / DETACHED_PROCESS on Windows), but the
    log path is hardcoded to `/tmp/ar_worker_<port>.log`, which is
    POSIX-only — Windows callers should run foreground (`--start`
    without `--bg`) and redirect stdout/stderr themselves.
  - `worker --stop` is POSIX-only: it shells out to `ss`/`lsof` to find
    the listening PID, reads `/proc/<pid>/cmdline` for the safety
    check, and sends `SIGTERM`/`SIGKILL` via `os.kill`. On Windows the
    detection commands are absent and the signals don't map — stop the
    daemon by killing the PID printed by `--start --bg` (Task Manager
    or `taskkill /PID <pid>`).

Prerequisites are the user's: activate a Python env where `fastapi +
uvicorn + pyyaml + torch` (plus torch_npu / triton / pandas / msprof /
nsys per DSL) are importable — ar_cli itself does not activate anything.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen


SCRIPTS_DIR = Path(__file__).resolve().parent   # .autoresearch/scripts/


def _worker_cfg() -> dict:
    """Pull `worker.{host,port}` from .autoresearch/config.yaml via
    utils.settings. Lazy + cached; safe before sys.argv parsing."""
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        from utils.settings import worker_defaults
        return worker_defaults()
    except Exception:
        return {"host": "0.0.0.0", "port": 9001}


# ---------------------------------------------------------------------------
# worker subcommand
# ---------------------------------------------------------------------------

def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _worker_env(args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    env["WORKER_BACKEND"] = args.backend
    env["WORKER_ARCH"] = args.arch
    env["WORKER_DEVICES"] = args.devices
    env["WORKER_PORT"] = str(args.port)
    env["WORKER_HOST"] = args.host
    return env


def _banner(args: argparse.Namespace, extra: Optional[dict] = None) -> str:
    rows = [
        ("Host", args.host), ("Backend", args.backend), ("Arch", args.arch),
        ("Devices", args.devices), ("Port", str(args.port)),
    ]
    if extra:
        rows.extend((k, str(v)) for k, v in extra.items())
    w = max(len(k) for k, _ in rows) + 1
    return "\n".join(f"  {k:<{w}}: {v}" for k, v in rows)


def _tail(path: Path, n: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError as e:
        return f"(cannot read {path}: {e})"


def _worker_start(args: argparse.Namespace) -> int:
    if args.bg:
        return _worker_start_daemon(args)

    print("=" * 48)
    print("AutoResearch Worker Service (foreground)")
    print("-" * 48)
    print(_banner(args))
    print("=" * 48, flush=True)
    os.environ.update({k: v for k, v in _worker_env(args).items()
                       if k.startswith("WORKER_")})
    sys.path.insert(0, str(SCRIPTS_DIR))
    from worker.server import start_server
    start_server(host=args.host, port=args.port)
    return 0


def _worker_start_daemon(args: argparse.Namespace) -> int:
    if _port_in_use(args.port):
        print(f"ERROR: port {args.port} is already in use. Stop the existing "
              f"daemon (python .autoresearch/scripts/ar_cli.py worker --stop "
              f"--port {args.port}) or pick another port.", file=sys.stderr)
        return 1

    log_path = Path(f"/tmp/ar_worker_{args.port}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab", buffering=0)

    cmd = [sys.executable, "-m", "worker.server"]
    popen_kwargs: dict = {
        "cwd": str(SCRIPTS_DIR),
        "env": _worker_env(args),
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": log_file,
        "close_fds": True,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED | NEW_PGROUP

    proc = subprocess.Popen(cmd, **popen_kwargs)
    log_file.close()

    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"ERROR: worker exited with code {proc.returncode}. "
                  f"Log tail:\n{_tail(log_path, 40)}", file=sys.stderr)
            return proc.returncode or 1
        if _port_in_use(args.port):
            break
        time.sleep(0.25)
    else:
        print(f"ERROR: worker PID {proc.pid} did not start listening on "
              f"{args.host}:{args.port} within 30s. Log tail:\n"
              f"{_tail(log_path, 40)}", file=sys.stderr)
        return 1

    print("=" * 48)
    print("AutoResearch Worker Service (daemon)")
    print("-" * 48)
    print(_banner(args, {"PID": proc.pid, "Log": log_path}))
    print("-" * 48)
    print(f"  Stop: python .autoresearch/scripts/ar_cli.py worker "
          f"--stop --port {args.port}")
    print("=" * 48)
    return 0


def _find_pid_on_port(port: int) -> Optional[int]:
    try:
        out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True,
                             check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        out = ""
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[3].endswith(f":{port}"):
            continue
        idx = fields[-1].find("pid=")
        if idx == -1:
            continue
        num = ""
        for ch in fields[-1][idx + 4:]:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            return int(num)

    try:
        out = subprocess.run(["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
                             capture_output=True, text=True,
                             check=False).stdout.strip()
        if out:
            return int(out.splitlines()[0])
    except FileNotFoundError:
        pass
    return None


def _cmdline_of(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        if raw:
            return raw.strip()
    except OSError:
        pass
    try:
        return subprocess.run(["ps", "-p", str(pid), "-o", "cmd="],
                              capture_output=True, text=True,
                              check=False).stdout.strip()
    except FileNotFoundError:
        return ""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _worker_stop(args: argparse.Namespace) -> int:
    pid = _find_pid_on_port(args.port)
    if pid is None:
        print(f"No process listening on port {args.port}.")
        return 0

    cmd = _cmdline_of(pid)
    if "worker.server" not in cmd and not args.force:
        print(f"ERROR: PID {pid} on port {args.port} does not look like an "
              f"autoresearch worker:\n  {cmd or '(cmdline unavailable)'}\n"
              f"Refusing to kill. Use --force, or `kill {pid}` manually.",
              file=sys.stderr)
        return 2

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"PID {pid} already gone.")
        return 0

    for _ in range(12):
        if not _alive(pid):
            print(f"Stopped PID {pid} (port {args.port}).")
            return 0
        time.sleep(0.25)

    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Force-killed PID {pid} (port {args.port}).")
    except ProcessLookupError:
        print(f"Stopped PID {pid} (port {args.port}).")
    except PermissionError as e:
        print(f"WARNING: cannot SIGKILL PID {pid}: {e}", file=sys.stderr)
        return 1
    return 0


def _worker_status(args: argparse.Namespace) -> int:
    url = f"http://{args.host}:{args.port}/api/v1/status"
    try:
        with urlopen(Request(url, method="GET"), timeout=5) as resp:
            body = resp.read().decode()
    except Exception as e:
        print(f"Worker on {args.host}:{args.port} unreachable: {e}",
              file=sys.stderr)
        return 1

    try:
        parsed = json.loads(body)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(body)
    return 0


def _cmd_worker(args: argparse.Namespace) -> int:
    # --start / --stop / --status are mutually exclusive; argparse enforces it.
    # `--host` default is action-dependent: bind on 0.0.0.0 for --start,
    # connect to 127.0.0.1 for --stop/--status (0.0.0.0 is bind-only and
    # cannot be the target of an outbound TCP connect).
    if args.host is None:
        args.host = (_worker_cfg()["host"] if args.start else "127.0.0.1")
    if args.start:
        return _worker_start(args)
    if args.stop:
        return _worker_stop(args)
    if args.status:
        return _worker_status(args)
    # No action flag given → show help for `worker`.
    print("worker: specify --start, --stop, or --status.\n"
          "Run `python .autoresearch/scripts/ar_cli.py worker --help` for "
          "details.", file=sys.stderr)
    return 2


def _add_worker_subcommand(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "worker",
        help="Manage the AutoResearch Worker Service (HTTP eval server).",
        description="Start / stop / check the AutoResearch Worker Service "
                    "on this machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mx = p.add_mutually_exclusive_group(required=False)
    mx.add_argument("--start", action="store_true",
                    help="Start the worker on this machine.")
    mx.add_argument("--stop", action="store_true",
                    help="Stop the daemon listening on --port. "
                         "POSIX-only (uses ss/lsof + SIGTERM/SIGKILL).")
    mx.add_argument("--status", action="store_true",
                    help="Curl /api/v1/status on --host:--port.")

    p.add_argument("--backend", required=True,
                   choices=["ascend", "cuda", "cpu"],
                   help="Hardware backend.")
    p.add_argument("--arch", required=True,
                   help="Arch string, e.g. ascend910b3 / a100 / x86_64.")
    p.add_argument("--devices", required=True,
                   help="Comma-separated device IDs, e.g. '2,5'.")
    # Worker daemon defaults come from .autoresearch/config.yaml:worker via
    # utils.settings; --start --host defaults override for the bind side,
    # --status / --stop side rewrites None → 127.0.0.1 in _cmd_worker.
    _w = _worker_cfg()
    p.add_argument("--host", default=None,
                   help="Bind / probe address. Defaults to config "
                        f"`worker.host` ({_w['host']}) for --start, "
                        "and 127.0.0.1 for --status / --stop (loopback "
                        "connect).")
    p.add_argument("--port", type=int, default=_w["port"],
                   help=f"TCP port (default: {_w['port']} — config "
                        "`worker.port`).")
    p.add_argument("--bg", action="store_true",
                   help="Daemon mode for --start. Detaches, logs to "
                        "/tmp/ar_worker_<port>.log, prints PID + log "
                        "path. The log path is POSIX-only; on Windows "
                        "run foreground (omit --bg) and redirect output.")
    p.add_argument("--force", action="store_true",
                   help="For --stop: skip the worker.server "
                        "cmdline safety check.")
    p.set_defaults(func=_cmd_worker)


# ---------------------------------------------------------------------------
# verify subcommand
# ---------------------------------------------------------------------------
#
# JSON-in / JSON-out wrapper around `task_config.run_eval`. Stable
# contract so external drivers can subprocess into us without coupling
# to internal Python APIs that may change.

_VERIFY_SENTINEL = "AR_VERIFY_RESULT:"


def _read_at_arg(value: str) -> str:
    """`@path` → file contents; anything else → returned verbatim."""
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


def _parse_task_config_text(text: str, hint_path: Optional[str] = None) -> dict:
    """Parse task config text as JSON, falling back to YAML.

    YAML accepted for parity with claude-autoresearch's on-disk task.yaml
    — internal callers (pipeline / baseline) can pass `@.../task.yaml`
    directly without re-serialising. JSON works for inline payloads and
    cross-tool integration.
    """
    text = text.lstrip("﻿")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # local import; only needed when JSON parse fails
        except ImportError as e:
            raise SystemExit(
                f"task-config parse failed: not JSON, and PyYAML unavailable "
                f"for YAML fallback ({e})")
        try:
            obj = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise SystemExit(
                f"task-config parse failed (tried JSON + YAML): {e}"
                + (f" — source: {hint_path}" if hint_path else ""))
        if not isinstance(obj, dict):
            raise SystemExit(
                f"task-config: expected dict, got {type(obj).__name__}"
                + (f" — source: {hint_path}" if hint_path else ""))
        return obj


def _taskconfig_from_dict(cfg: dict) -> Any:
    """Build a `TaskConfig` from a parsed dict — same shape as task.yaml."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    from task_config.loader import TaskConfig
    name = cfg.get("name") or cfg.get("op_name")
    if not name:
        raise SystemExit("task-config: missing required field 'name' (or 'op_name')")

    eval_block = cfg.get("eval", {}) if isinstance(cfg.get("eval"), dict) else {}
    metric_block = cfg.get("metric", {}) if isinstance(cfg.get("metric"), dict) else {}
    smoke_block = cfg.get("smoke_test", {}) if isinstance(cfg.get("smoke_test"), dict) else {}
    agent_block = cfg.get("agent", {}) if isinstance(cfg.get("agent"), dict) else {}
    cc_block = cfg.get("code_checker", {}) if isinstance(cfg.get("code_checker"), dict) else {}
    worker_block = cfg.get("worker", {}) if isinstance(cfg.get("worker"), dict) else {}

    # Flat / inline aliases — accept top-level eval_timeout /
    # warmup_times / run_times alongside the nested task.yaml `eval`
    # block so inline JSON payloads stay terse.
    eval_timeout = cfg.get("eval_timeout", eval_block.get("timeout", 600))
    warmup_times = cfg.get("warmup_times", eval_block.get("warmup_times", 10))
    run_times = cfg.get("run_times", eval_block.get("run_times", 100))

    worker_urls = worker_block.get("urls") or cfg.get("worker_urls") or []
    if isinstance(worker_urls, str):
        worker_urls = [u.strip() for u in worker_urls.split(",") if u.strip()]

    devices_raw = cfg.get("devices", [])
    if isinstance(devices_raw, int):
        devices = [devices_raw]
    elif isinstance(devices_raw, str):
        devices = [int(d.strip()) for d in devices_raw.split(",") if d.strip()]
    elif isinstance(devices_raw, list):
        devices = [int(d) for d in devices_raw]
    else:
        devices = []

    constraints = {}
    for metric_name, spec in (cfg.get("constraints") or {}).items():
        if isinstance(spec, dict):
            constraints[metric_name] = (spec["op"], spec["value"])
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            constraints[metric_name] = tuple(spec)

    # Default editable_files to `["kernel.py"]` when the caller didn't
    # declare one — `_gen_eval_script` indexes `editable_files[0]` to
    # decide which file to import the kernel symbol from, and a TaskConfig
    # built straight from inline JSON wouldn't have it set otherwise.
    # scaffold-built task.yaml always populates this; we mirror that
    # contract here.
    editable_files = cfg.get("editable_files") or ["kernel.py"]

    return TaskConfig(
        name=name,
        description=cfg.get("description", ""),
        dsl=cfg.get("dsl"),
        framework=cfg.get("framework"),
        backend=cfg.get("backend"),
        arch=cfg.get("arch"),
        editable_files=editable_files,
        ref_file=cfg.get("ref_file") or agent_block.get("ref_file") or "reference.py",
        eval_timeout=int(eval_timeout),
        warmup_times=int(warmup_times),
        run_times=int(run_times),
        primary_metric=metric_block.get("primary", "score"),
        lower_is_better=bool(metric_block.get("lower_is_better", True)),
        improvement_threshold=float(metric_block.get("improvement_threshold", 0.0)),
        constraints=constraints,
        smoke_test_script=smoke_block.get("script"),
        smoke_test_timeout=int(smoke_block.get("timeout", 10)),
        code_checker_enabled=bool(cc_block.get("enabled", True)),
        max_rounds=int(agent_block.get("max_rounds", 30)),
        worker_urls=worker_urls,
        devices=devices,
    )


def _materialize_task_dir(config: Any, impl_code: str, ref_code: str,
                          source_task_dir: Optional[str]) -> str:
    """Lay out a tempdir that `run_eval` / `_build_package` can consume.

    Writes kernel.py + <config.ref_file>; copies extra editable_files and
    sibling .py support files plus `.ar_state/progress.json` (for sticky
    baseline override) from `source_task_dir` when given. Caller owns the
    tempdir and is responsible for cleanup.
    """
    tmp = tempfile.mkdtemp(prefix="ar_verify_")
    Path(tmp, "kernel.py").write_text(impl_code, encoding="utf-8")
    Path(tmp, config.ref_file).write_text(ref_code, encoding="utf-8")

    if source_task_dir and os.path.isdir(source_task_dir):
        # editable_files (besides kernel.py — we already have impl_code).
        for fname in config.editable_files:
            if fname in ("kernel.py", config.ref_file):
                continue
            src = os.path.join(source_task_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmp, fname))

        # Other .py support files that the user dropped next to the kernel.
        for f in os.listdir(source_task_dir):
            if (f.endswith(".py")
                    and f not in {"kernel.py", config.ref_file}
                    and f not in config.editable_files
                    and not f.startswith(".")):
                src = os.path.join(source_task_dir, f)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(tmp, f))

        # `.ar_state/progress.json` carries the sticky baseline anchor
        # eval_request looks up. Without it sticky never kicks in and
        # every round re-measures ref from scratch.
        src_state = os.path.join(source_task_dir, ".ar_state")
        if os.path.isdir(src_state):
            dst_state = os.path.join(tmp, ".ar_state")
            os.makedirs(dst_state, exist_ok=True)
            for f in ("progress.json",):
                sp = os.path.join(src_state, f)
                if os.path.isfile(sp):
                    shutil.copy2(sp, os.path.join(dst_state, f))

    return tmp


def _filter_by_mode(eval_dict: dict, mode: str) -> dict:
    """Strip fields not requested by mode. Keeps the response schema
    matching the documented `verify-only` / `profile-only` semantics
    even though the underlying eval always runs both phases."""
    if mode == "verify-only":
        eval_dict = dict(eval_dict)
        eval_dict["metrics"] = {}
    elif mode == "profile-only":
        eval_dict = dict(eval_dict)
        # Treat correctness as a pass-through — caller asked for profile.
        eval_dict.setdefault("correctness", True)
    return eval_dict


def _emit_verify_result(payload: dict) -> int:
    """Print sentinel-tagged JSON and return rc=0 whenever a result is
    emitted — `kernel_fail` and `infra_fail` are still informational
    outcomes the caller reads off the JSON, not CLI-level errors.

    Reserving non-zero for true CLI failures matches the contract the
    eval-wrapper layer used to expose: stdout-JSON-or-bust. baseline.py
    in particular aborts on any non-zero rc *before* parsing the JSON
    body, so distinguishing ok vs kernel_fail at exit-code level would
    deadlock the BASELINE → PLAN transition for any seed kernel that
    happened to fail correctness on round 0 (which is a routine,
    expected path — first plan items just rewrite the seed).
    """
    sys.stdout.write("\n" + _VERIFY_SENTINEL
                     + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


def _verify_infra_fail(msg: str) -> dict:
    return {
        "outcome": "infra_fail",
        "correctness": False,
        "metrics": {},
        "error": msg,
        "error_source": None,
    }


def _cmd_verify(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        cfg_text = _read_at_arg(args.task_config)
        hint = args.task_config[1:] if args.task_config.startswith("@") else None
        cfg_dict = _parse_task_config_text(cfg_text, hint_path=hint)
        config = _taskconfig_from_dict(cfg_dict)
        impl_code = _read_at_arg(args.impl)
        ref_code = _read_at_arg(args.reference)
    except SystemExit as e:
        return _emit_verify_result(_verify_infra_fail(str(e)))
    except Exception as e:
        return _emit_verify_result(_verify_infra_fail(
            f"verify args parse failed: {type(e).__name__}: {e}"))

    worker_urls = None
    if args.worker_url:
        worker_urls = [u.strip() for u in args.worker_url.split(",") if u.strip()]

    tmp_dir = None
    try:
        tmp_dir = _materialize_task_dir(config, impl_code, ref_code,
                                        source_task_dir=args.task_dir)
        from task_config.eval_client import run_eval
        from utils.failure_extractor import extract_failure_signals
        result = run_eval(tmp_dir, config,
                          device_id=args.device_id,
                          worker_urls=worker_urls)
    except Exception as e:
        return _emit_verify_result(_verify_infra_fail(
            f"run_eval failed: {type(e).__name__}: {e}\n"
            f"{traceback.format_exc()[-2000:]}"))
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    payload = {
        "outcome": result.outcome.value,
        "correctness": result.correctness,
        "metrics": result.metrics or {},
        "error": result.error,
        "error_source": result.error_source,
    }
    if not result.correctness or result.error:
        payload["failure_signals"] = extract_failure_signals(
            result.raw_output).to_dict()
        payload["raw_output_tail"] = (result.raw_output or "")[-4000:]
    payload = _filter_by_mode(payload, args.mode)
    return _emit_verify_result(payload)


def _add_verify_subcommand(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "verify",
        help="Run verify + profile against a kernel (JSON-in / JSON-out).",
        description=(
            "Materialise impl + reference into a tempdir, invoke run_eval, "
            "emit a sentinel-tagged JSON result."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task-config", required=True,
                   help="@path|inline YAML or JSON. Required fields: name "
                        "(or op_name). Other fields mirror task.yaml.")
    p.add_argument("--impl", required=True,
                   help="@path or inline kernel.py source.")
    p.add_argument("--reference", required=True,
                   help="@path or inline reference source.")
    p.add_argument("--task-dir", default=None,
                   help="Optional: source directory for editable_files / "
                        "extra .py / .ar_state/progress.json (sticky "
                        "baseline). Without it, the materialised tempdir "
                        "contains only kernel.py + ref_file.")
    p.add_argument("--device-id", type=int, default=None,
                   help="Local device id for the eval subprocess.")
    p.add_argument("--worker-url", default=None,
                   help="Remote worker URL(s), comma-separated. Overrides "
                        "task-config worker.urls.")
    p.add_argument("--mode", default="verify+profile",
                   choices=["verify+profile", "verify-only", "profile-only"],
                   help="Output filter; eval always runs full pipeline today.")
    p.set_defaults(func=_cmd_verify)


# ---------------------------------------------------------------------------
# env subcommand
# ---------------------------------------------------------------------------
#
# JSON-in / JSON-out env introspection. Centralises hardware probes and
# the DSL→backend table so external callers can ask "what's available"
# / "what DSLs do you accept" without import-spelunking through
# verifier.adapters.factory.

_ENV_SENTINEL = "AR_ENV_RESULT:"


def _emit_env_result(payload: dict, ok: bool = True) -> int:
    sys.stdout.write("\n" + _ENV_SENTINEL
                     + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0 if ok else 1


def _try_import_version(module_name: str) -> Optional[str]:
    try:
        import importlib
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(mod, "__version__", "unknown")


def _detect_cuda() -> Optional[dict]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except Exception:
        return None
    try:
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "id": i,
                "name": props.name,
                "arch": _cuda_arch_token(props.name),
                "compute_cap": f"{props.major}.{props.minor}",
                "total_memory_mb": props.total_memory // (1024 * 1024),
            })
        arch = devices[0]["arch"] if devices else None
        return {"available": True, "arch": arch, "devices": devices}
    except Exception as e:
        return {"available": False, "error": str(e)}


def _cuda_arch_token(name: str) -> Optional[str]:
    norm = (name or "").lower().replace(" ", "").replace("-", "")
    for token in ("a100", "h100", "a800", "h800", "v100", "t4",
                  "rtx4090", "rtx3090", "l40", "l4"):
        if token in norm:
            return token
    return name.lower() or None


def _detect_ascend() -> Optional[dict]:
    try:
        r = subprocess.run(["npu-smi", "info"], capture_output=True,
                           text=True, timeout=15)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    info: dict = {"available": True}
    devices: list = []
    arch: Optional[str] = None
    try:
        import torch
        import torch_npu  # noqa: F401
        info["torch_npu_loaded"] = True
        try:
            for i in range(torch.npu.device_count()):
                nm = torch.npu.get_device_name(i)
                devices.append({"id": i, "name": nm,
                                "arch": _ascend_arch_token(nm)})
            if devices:
                arch = devices[0]["arch"]
        except Exception as e:
            info["device_probe_error"] = str(e)
    except Exception as e:
        info["torch_npu_loaded"] = False
        info["torch_npu_error"] = str(e)
        # Fallback: parse npu-smi's main table.
        import re
        for m in re.finditer(r"^\|\s*(\d+)\s+(\S+)\s*\|", r.stdout,
                              re.MULTILINE):
            raw = m.group(2).strip().lower()
            devices.append({"id": int(m.group(1)), "name": raw,
                            "arch": _ascend_arch_token(raw)})
        if devices:
            arch = devices[0]["arch"]
    info["devices"] = devices
    info["arch"] = arch
    return info


def _ascend_arch_token(name: str) -> Optional[str]:
    if not name:
        return None
    lower = name.lower()
    return lower if lower.startswith("ascend") else f"ascend{lower}"


def _detect_cpu() -> dict:
    import platform
    arch = platform.machine().lower()
    return {
        "available": True, "arch": arch,
        "devices": [{"id": 0, "name": "cpu", "arch": arch}],
        "python_impl": platform.python_implementation(),
    }


def _fetch_worker_status(worker_url: str, timeout: float = 5.0) -> dict:
    url = worker_url.strip()
    if not url.startswith("http"):
        url = f"http://{url}"
    url = url.rstrip("/") + "/api/v1/status"
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return {"_error": f"failed to GET {url}: {e}"}
    if not isinstance(data, dict):
        return {"_error": f"non-dict response from {url}"}
    return data


def _backend_device_type(backend: str) -> str:
    return {"ascend": "npu", "cuda": "cuda", "cpu": "cpu"}.get(
        backend.lower(), backend.lower())


def _list_dsl_table() -> list[dict]:
    """Pull the canonical DSL→backend mapping from the verifier factory.

    Single source of truth; mirrors what `get_dsl_adapter` accepts. Each
    adapter declares its backend via `default_backend()`; we layer
    `device_type` on top per the backend→device-type map above.
    """
    sys.path.insert(0, str(SCRIPTS_DIR))
    from verifier.adapters.factory import list_dsls, get_dsl_adapter
    out = []
    for name in sorted(list_dsls()):
        try:
            backend = get_dsl_adapter(name).default_backend()
        except Exception:
            backend = ""
        out.append({"name": name, "backend": backend,
                    "device_type": _backend_device_type(backend) if backend else ""})
    return out


def _cmd_env_detect(args: argparse.Namespace) -> int:
    if args.worker_url:
        status = _fetch_worker_status(args.worker_url)
        if "_error" in status:
            return _emit_env_result(
                {"remote": True, "url": args.worker_url, "ok": False,
                 "error": status["_error"]}, ok=False)
        return _emit_env_result(
            {"remote": True, "url": args.worker_url, "ok": True, **status},
            ok=True)
    payload = {
        "remote": False,
        "backends": {
            "cuda": _detect_cuda(),
            "ascend": _detect_ascend(),
            "cpu": _detect_cpu(),
        },
        "sdks": {
            "torch": _try_import_version("torch"),
            "torch_npu": _try_import_version("torch_npu"),
            "mindspore": _try_import_version("mindspore"),
            "triton": _try_import_version("triton"),
            "numpy": _try_import_version("numpy"),
        },
    }
    return _emit_env_result(payload, ok=True)


def _cmd_env_check(args: argparse.Namespace) -> int:
    """Return whether (framework, backend, dsl) is runnable here.

    The check is intentionally light — it confirms the DSL is registered
    and (for local runs) the backend's hardware is reachable. Deep SDK
    version checks live in user-supplied scripts; we surface what we know.
    """
    issues: list[str] = []
    try:
        from verifier.adapters.factory import get_dsl_adapter
        try:
            adapter = get_dsl_adapter(args.dsl)
        except Exception as e:
            issues.append(f"unknown DSL {args.dsl!r}: {e}")
            adapter = None
        if adapter is not None:
            declared = adapter.default_backend()
            if args.backend.lower() != declared.lower():
                issues.append(
                    f"backend mismatch: --backend={args.backend!r} but "
                    f"DSL {args.dsl!r} expects {declared!r}")
    except Exception as e:
        issues.append(f"verifier registry import failed: {e}")

    if not args.remote:
        # Validate that the local backend is actually present.
        if args.backend.lower() == "cuda":
            if _detect_cuda() is None:
                issues.append("backend=cuda but no CUDA device detected")
        elif args.backend.lower() == "ascend":
            if _detect_ascend() is None:
                issues.append("backend=ascend but npu-smi probe failed")

    ok = not issues
    return _emit_env_result({
        "ok": ok, "framework": args.framework, "backend": args.backend,
        "dsl": args.dsl, "arch": args.arch, "issues": issues,
    }, ok=ok)


def _cmd_env_list_dsls(args: argparse.Namespace) -> int:
    try:
        dsls = _list_dsl_table()
    except Exception as e:
        return _emit_env_result(
            {"ok": False, "error": f"DSL registry import failed: {e}"},
            ok=False)
    return _emit_env_result({"dsls": dsls}, ok=True)


def _cmd_env(args: argparse.Namespace) -> int:
    handler = getattr(args, "env_func", None)
    if handler is None:
        print("env: specify a subcommand (detect | check | list-dsls).\n"
              "Run `python .autoresearch/scripts/ar_cli.py env --help` for "
              "details.", file=sys.stderr)
        return 2
    return handler(args)


def _add_env_subcommand(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "env",
        help="Environment introspection (detect / check / list-dsls).",
        description="JSON-in / JSON-out env queries. Sentinel: AR_ENV_RESULT:",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_cmd_env)
    env_sub = p.add_subparsers(dest="env_command",
                               metavar="{detect,check,list-dsls}")

    pd = env_sub.add_parser("detect",
                            help="Probe local backends + SDKs, or fetch "
                                 "/api/v1/status from a remote worker.")
    pd.add_argument("--worker-url", default=None,
                    help="If given, GET worker status instead of local probe.")
    pd.set_defaults(env_func=_cmd_env_detect)

    pc = env_sub.add_parser("check",
                            help="Validate (framework, backend, dsl). "
                                 "Exit 0 on ok, 1 on issues.")
    pc.add_argument("--framework", required=True,
                    help="torch | mindspore | numpy")
    pc.add_argument("--backend", required=True,
                    help="cuda | ascend | cpu")
    pc.add_argument("--dsl", required=True,
                    help="e.g. triton_ascend, triton_cuda, tilelang_cuda, ...")
    pc.add_argument("--arch", default=None,
                    help="Informational; recorded in the response.")
    pc.add_argument("--remote", action="store_true",
                    help="Skip local hardware probing (you're running "
                         "against a remote worker).")
    pc.set_defaults(env_func=_cmd_env_check)

    pl = env_sub.add_parser("list-dsls",
                            help="Emit the DSL → backend table the "
                                 "verifier accepts.")
    pl.set_defaults(env_func=_cmd_env_list_dsls)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        prog="ar_cli",
        description="AutoResearch CLI. Subcommands: worker, verify, env.",
    )
    sub = p.add_subparsers(dest="command", metavar="{worker,verify,env}")
    _add_worker_subcommand(sub)
    _add_verify_subcommand(sub)
    _add_env_subcommand(sub)

    args = p.parse_args()
    if not args.command:
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
