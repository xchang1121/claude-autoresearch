"""Remote worker dispatch through SSH and local ssh -L tunnels."""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from typing import Optional

from .diagnostics import classify, has_fatal, render_findings
from .remote_env import source_env_script_bash
from .remote_probe import probe_remote
from .tunnel import kill_pid_hint, tunnel_start, tunnel_stop_silent, who_holds_port
from .worker_config import WorkerConfig, WorkerTiming, parse_devices
from .worker_service import curl_health, curl_status, is_ready


def load_remote_host_config(alias: str, config_path: Optional[str]) -> Optional[dict]:
    return WorkerConfig.load(config_path).host(alias)


def load_default_port(config_path: Optional[str]) -> Optional[int]:
    return WorkerConfig.load(config_path).port


def _step(message: str) -> None:
    print(f"[ar_cli] {message}", file=sys.stderr, flush=True)


def _device_ids_from_arg(devices: Optional[str]) -> Optional[list[int]]:
    if devices is None:
        return None
    return parse_devices(devices)


def _build_remote_start_cmd(
    host_cfg: dict,
    *,
    backend: str,
    arch: str,
    devices: str,
    port: int,
    timing: WorkerTiming,
) -> str:
    repo_path = host_cfg["repo_path"]
    env_script = host_cfg.get("env_script")
    parts = [source_env_script_bash(env_script)]
    parts.append(f"export PYTHONPATH={shlex.quote(repo_path)}/scripts:${{PYTHONPATH:-}}")
    parts.append("export WORKER_HOST=127.0.0.1")
    parts.append("export AR_CLI_QUIET=1")
    parts.append(f"export AR_WORKER_READY_TIMEOUT={timing.ready_timeout}")
    parts.append(f"export AR_WORKER_READY_POLL_INTERVAL={timing.ready_poll_interval}")
    parts.append(f"export AR_WORKER_READY_PROBE_TIMEOUT={timing.ready_probe_timeout}")
    parts.append(f"cd {shlex.quote(repo_path)}")
    parts.append(
        " ".join(
            [
                "python",
                "scripts/ar_cli.py",
                "worker",
                "--start",
                "--backend",
                shlex.quote(backend),
                "--arch",
                shlex.quote(arch),
                "--devices",
                shlex.quote(devices),
                "--port",
                str(port),
            ]
        )
    )
    return "\n".join(parts)


def _ssh_dispatch(ssh_alias: str, bash_cmd: str) -> int:
    return subprocess.call(["ssh", "-o", "LogLevel=ERROR", ssh_alias, f"bash -lc {shlex.quote(bash_cmd)}"])


def dispatch_start(
    alias: str,
    host_cfg: dict,
    backend: Optional[str],
    arch: Optional[str],
    devices: Optional[str],
    port: int,
    dsl: Optional[str] = None,
) -> int:
    if "repo_path" not in host_cfg:
        _step(f"remote_worker.hosts.{alias} missing repo_path")
        return 2

    ssh_alias = host_cfg.get("ssh_alias") or alias
    repo_path = host_cfg.get("repo_path")
    env_script = host_cfg.get("env_script")
    log_file = f"/tmp/akg_worker_{port}.log"

    cfg = WorkerConfig.load(None)
    effective_backend = (backend or cfg.backend).strip().lower()
    effective_dsl = dsl or cfg.dsl
    effective_devices = devices or cfg.devices
    timing = cfg.timing
    probe_devices = _device_ids_from_arg(effective_devices)

    _step(f"[1/4] probing 127.0.0.1:{port}/api/v1/status ...")
    status = curl_status("127.0.0.1", port, timeout=timing.status_timeout)
    if is_ready(status):
        _step("[1/4] daemon already ready")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0

    _step(f"[2/4] rebuilding ssh -L :{port} -> {ssh_alias} ...")
    tunnel_stop_silent(port, ssh_alias)
    pid = tunnel_start(ssh_alias, port)
    if not pid:
        _step("[2/4] tunnel start failed; running remote diagnostics")
        facts = probe_remote(ssh_alias, env_script, port, log_file, repo_path, probe_devices, effective_backend)
        render_findings(classify(facts, port, backend=effective_backend, dsl=effective_dsl, for_start=True), facts.get("LOG_TAIL", ""))
        return 1
    _step(f"[2/4] tunnel pid={pid}; probing status again ...")
    status = curl_status("127.0.0.1", port, timeout=timing.status_timeout)
    if is_ready(status):
        _step("[2/4] tunnel was stale; remote daemon is already ready")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0

    _step("[3/4] remote diagnostics ...")
    facts = probe_remote(ssh_alias, env_script, port, log_file, repo_path, probe_devices, effective_backend)
    findings = classify(facts, port, backend=effective_backend, dsl=effective_dsl, for_start=True)
    if has_fatal(findings):
        render_findings(findings, facts.get("LOG_TAIL", ""))
        return 1

    if arch is None:
        if effective_backend == "ascend":
            arch = (facts.get("ARCH") or "").strip().lower()
        elif effective_backend == "cuda":
            arch = (facts.get("CUDA_ARCH") or "").strip().lower()
        elif effective_backend == "cpu":
            arch = (facts.get("CPU_ARCH") or "").strip().lower()
    if not arch:
        _step("[3/4] could not derive arch; pass --arch explicitly")
        render_findings(findings, facts.get("LOG_TAIL", ""))
        return 1

    _step(f"[3/4] probe OK: backend={effective_backend}, arch={arch}, devices={effective_devices}, dsl={effective_dsl or '(any)'}")
    _step(f"[4/4] starting remote daemon on {ssh_alias}:{port} ...")
    remote_cmd = _build_remote_start_cmd(
        host_cfg,
        backend=effective_backend,
        arch=arch,
        devices=effective_devices,
        port=port,
        timing=timing,
    )
    rc = _ssh_dispatch(ssh_alias, remote_cmd)
    if rc != 0:
        _step(f"[4/4] remote start rc={rc}; diagnostics follow")
        facts2 = probe_remote(ssh_alias, env_script, port, log_file, repo_path, probe_devices, effective_backend)
        render_findings(classify(facts2, port, backend=effective_backend, dsl=effective_dsl, for_start=True), facts2.get("LOG_TAIL", ""))
        return rc

    _step(f"[4/4] polling /status ready for up to {timing.ready_timeout}s ...")
    deadline = time.time() + timing.ready_timeout
    last = time.time()
    while time.time() < deadline:
        status = curl_status("127.0.0.1", port, timeout=timing.ready_probe_timeout)
        if is_ready(status):
            _step("[4/4] ready")
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return 0
        now = time.time()
        if now - last >= timing.ready_poll_interval:
            elapsed = int(now - (deadline - timing.ready_timeout))
            _step(f"   waiting ({elapsed}s/{timing.ready_timeout}s)")
            last = now
        time.sleep(1)

    facts3 = probe_remote(ssh_alias, env_script, port, log_file, repo_path, probe_devices, effective_backend)
    render_findings(classify(facts3, port, backend=effective_backend, dsl=effective_dsl, for_start=True), facts3.get("LOG_TAIL", ""))
    return 1


def dispatch_stop(alias: str, host_cfg: dict, port: int) -> int:
    ssh_alias = host_cfg.get("ssh_alias") or alias
    tunnel_stop_silent(port, ssh_alias)
    print(f"[ar_cli] tore down local tunnel for :{port}")
    rc = _ssh_dispatch(ssh_alias, f"lsof -ti :{port} | xargs -r kill")
    if rc != 0:
        print(f"[ar_cli] remote daemon stop rc={rc}", file=sys.stderr)
        return rc
    print(f"[ar_cli] killed remote daemon on {ssh_alias}:{port}")
    return 0


def dispatch_status(alias: str, host_cfg: dict, port: int, *, backend: Optional[str] = None, dsl: Optional[str] = None) -> int:
    status = curl_status("127.0.0.1", port)
    if status is None:
        holder = who_holds_port(port)
        if holder is None:
            print(f"Worker 127.0.0.1:{port} unreachable; local port is free. Run --start.")
        else:
            print(
                f"Worker 127.0.0.1:{port} unreachable; local port is held by PID={holder['pid']}\n"
                f"  cmdline: {holder['cmdline'][:160]}\n"
                f"  hint: {kill_pid_hint(holder['pid'])}"
            )
        ssh_alias = host_cfg.get("ssh_alias") or alias
        if ssh_alias != "local":
            cfg = WorkerConfig.load(None)
            effective_backend = backend or cfg.backend
            effective_dsl = dsl or cfg.dsl
            facts = probe_remote(
                ssh_alias,
                host_cfg.get("env_script"),
                port,
                f"/tmp/akg_worker_{port}.log",
                host_cfg.get("repo_path"),
                None,
                effective_backend,
            )
            render_findings(classify(facts, port, backend=effective_backend, dsl=effective_dsl, for_start=False), facts.get("LOG_TAIL", ""))
        return 1

    health = curl_health("127.0.0.1", port)
    out = dict(status)
    if health is not None:
        out["health"] = {
            "healthy": bool(health.get("healthy")),
            "probed_device": health.get("probed_device"),
            "free": health.get("free"),
            "note": health.get("note"),
            "error": health.get("error"),
        }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if health is not None and not health.get("healthy"):
        print(f"[ar_cli] /status OK but /health degraded: {health.get('error')!r}", file=sys.stderr)
        return 1
    return 0


def dispatch_reconnect_tunnel(alias: str, host_cfg: dict, port: int) -> int:
    ssh_alias = host_cfg.get("ssh_alias") or alias
    tunnel_stop_silent(port, ssh_alias)
    pid = tunnel_start(ssh_alias, port)
    if pid:
        print(f"[ar_cli] ssh tunnel reconnected pid={pid}")
    status = curl_status("127.0.0.1", port)
    if status is None:
        print("[ar_cli] status still unreachable after reconnect", file=sys.stderr)
        return 1
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0
