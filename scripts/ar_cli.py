#!/usr/bin/env python3
"""AutoResearch command line entry."""
from __future__ import annotations

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from cli_service import remote_dispatch
from cli_service.worker_config import WorkerConfig, parse_devices, probe_local_arch
from cli_service.worker_service import WorkerService

_LOGO = r"""
    _         _        ____                               _
   / \  _   _| |_ ___ |  _ \ ___  ___  ___  __ _ _ __ ___| |__
  / _ \| | | | __/ _ \| |_) / _ \/ __|/ _ \/ _` | '__/ __| '_ \
 / ___ \ |_| | || (_) |  _ <  __/\__ \  __/ (_| | | | (__| | | |
/_/   \_\__,_|\__\___/|_| \_\___||___/\___|\__,_|_|  \___|_| |_|
"""


def _print_logo_once() -> None:
    if os.environ.get("AR_CLI_QUIET") == "1":
        return
    print(_LOGO.rstrip("\n"))
    print("AutoResearch Worker")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ar_cli", description="AutoResearch CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    worker = sub.add_parser("worker", help="Start, stop, or check the worker daemon.")
    action = worker.add_mutually_exclusive_group(required=True)
    action.add_argument("--start", action="store_true", help="Start the worker daemon.")
    action.add_argument("--stop", action="store_true", help="Stop the worker daemon.")
    action.add_argument("--status", action="store_true", help="Probe /api/v1/status and /api/v1/health.")
    worker.add_argument("--backend", choices=["ascend", "cuda", "cpu"], help="Hardware backend.")
    worker.add_argument("--arch", help="Hardware arch token. Auto-derived when omitted.")
    worker.add_argument("--devices", help="Comma-separated device IDs, e.g. 0,1.")
    worker.add_argument("--dsl", help="Target DSL for diagnostics, e.g. triton_ascend or ascendc.")
    worker.add_argument("--port", type=int, help="Worker TCP port. Defaults to config.yaml worker.port.")
    worker.add_argument("--remote-host", help="SSH alias under config.yaml remote_worker.hosts.")
    worker.set_defaults(func=_worker_cmd)
    return parser


def _port_or_default(args, cfg: WorkerConfig) -> int:
    port = args.port if args.port is not None else cfg.port
    if port <= 0 or port > 65535:
        raise ValueError(f"--port out of range: {port}")
    return port


def _env_or_arg(name: str, arg_value: str | None) -> str | None:
    return (arg_value or os.environ.get(name) or "").strip() or None


def _worker_cmd(args) -> int:
    cfg = WorkerConfig.load(None)
    try:
        port = _port_or_default(args, cfg)
    except ValueError as exc:
        print(f"[ar_cli] {exc}", file=sys.stderr)
        return 2

    if args.start:
        _print_logo_once()

    if args.remote_host:
        host_cfg = remote_dispatch.load_remote_host_config(args.remote_host, None)
        if host_cfg is None:
            print(f"[ar_cli] missing remote_worker.hosts.{args.remote_host} in ./config.yaml", file=sys.stderr)
            return 2
        if args.status:
            backend = _env_or_arg("WORKER_BACKEND", args.backend) or cfg.backend
            dsl = _env_or_arg("WORKER_DSL", args.dsl) or cfg.dsl
            return remote_dispatch.dispatch_status(args.remote_host, host_cfg, port, backend=backend, dsl=dsl)
        if args.stop:
            return remote_dispatch.dispatch_stop(args.remote_host, host_cfg, port)
        backend = _env_or_arg("WORKER_BACKEND", args.backend)
        arch = _env_or_arg("WORKER_ARCH", args.arch)
        devices = _env_or_arg("WORKER_DEVICES", args.devices)
        dsl = _env_or_arg("WORKER_DSL", args.dsl)
        return remote_dispatch.dispatch_start(
            args.remote_host,
            host_cfg,
            backend=backend,
            arch=arch,
            devices=devices,
            port=port,
            dsl=dsl,
        )

    if args.status:
        backend = _env_or_arg("WORKER_BACKEND", args.backend) or cfg.backend
        dsl = _env_or_arg("WORKER_DSL", args.dsl) or cfg.dsl
        return remote_dispatch.dispatch_status("local", {"ssh_alias": "local"}, port, backend=backend, dsl=dsl)

    service = WorkerService()
    if args.stop:
        return service.stop(port=port)

    backend = (_env_or_arg("WORKER_BACKEND", args.backend) or cfg.backend).lower()
    devices_raw = _env_or_arg("WORKER_DEVICES", args.devices) or cfg.devices
    try:
        devices = parse_devices(devices_raw)
    except ValueError as exc:
        print(f"[ar_cli] {exc}", file=sys.stderr)
        return 2
    arch = _env_or_arg("WORKER_ARCH", args.arch) or probe_local_arch(backend, devices[0]) or cfg.arch

    os.environ.setdefault("AR_WORKER_READY_TIMEOUT", str(cfg.timing.ready_timeout))
    os.environ.setdefault("AR_WORKER_READY_POLL_INTERVAL", str(cfg.timing.ready_poll_interval))
    os.environ.setdefault("AR_WORKER_READY_PROBE_TIMEOUT", str(cfg.timing.ready_probe_timeout))
    host = (os.environ.get("WORKER_HOST") or "0.0.0.0").strip()
    return service.start(backend=backend, arch=arch, devices=devices, host=host, port=port)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
