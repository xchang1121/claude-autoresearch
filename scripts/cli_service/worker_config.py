"""Worker CLI configuration and local architecture probes."""
from __future__ import annotations

import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from eval.arch_normalize import (
    normalize_ascend_arch_name,
    normalize_cpu_arch_name,
    normalize_cuda_arch_name,
)


@dataclass(frozen=True)
class WorkerTiming:
    ready_timeout: float = 60.0
    ready_poll_interval: float = 5.0
    ready_probe_timeout: float = 3.0
    status_timeout: float = 3.0


@dataclass(frozen=True)
class WorkerConfig:
    port: int = 9001
    backend: str = "cuda"
    arch: str = "a100"
    devices: str = "0"
    dsl: Optional[str] = None
    hosts: Dict[str, dict] = field(default_factory=dict)
    timing: WorkerTiming = field(default_factory=WorkerTiming)
    source_path: Optional[str] = None

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "WorkerConfig":
        resolved = _resolve(config_path)
        if resolved is None:
            return cls()
        data = _load_yaml(resolved)
        if data is None:
            return cls(source_path=resolved)

        worker = data.get("worker") or {}
        defaults = data.get("defaults") or {}
        hosts = ((data.get("remote_worker") or {}).get("hosts") or {})

        td = WorkerTiming()
        return cls(
            port=_int_in_range(worker.get("port"), 1, 65535, cls.port),
            backend=_str_or(defaults.get("backend"), cls.backend).lower(),
            arch=_str_or(defaults.get("arch"), cls.arch),
            devices=_str_or(defaults.get("devices"), cls.devices),
            dsl=_optional_str(defaults.get("dsl")),
            hosts=dict(hosts),
            timing=WorkerTiming(
                ready_timeout=_float(worker.get("ready_timeout"), td.ready_timeout),
                ready_poll_interval=_float(
                    worker.get("ready_poll_interval"), td.ready_poll_interval
                ),
                ready_probe_timeout=_float(
                    worker.get("ready_probe_timeout"), td.ready_probe_timeout
                ),
                status_timeout=_float(worker.get("status_timeout"), td.status_timeout),
            ),
            source_path=resolved,
        )

    def host(self, alias: str) -> Optional[dict]:
        return self.hosts.get(alias)


def _resolve(config_path: Optional[str]) -> Optional[str]:
    if config_path is None:
        default = Path.cwd() / "config.yaml"
        return str(default) if default.is_file() else None
    return config_path if Path(config_path).is_file() else None


def _load_yaml(config_path: str) -> Optional[dict]:
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[ar_cli] failed to read {config_path}: {exc}", file=sys.stderr)
        return None


def _float(val, default: float) -> float:
    if isinstance(val, (int, float)) and val > 0:
        return float(val)
    return default


def _int_in_range(val, lo: int, hi: int, default: int) -> int:
    if isinstance(val, int) and lo <= val <= hi:
        return val
    return default


def _str_or(val, default: str) -> str:
    return str(val).strip() if isinstance(val, str) and str(val).strip() else default


def _optional_str(val) -> Optional[str]:
    text = str(val).strip() if isinstance(val, str) else ""
    return text or None


def parse_devices(raw: Optional[str]) -> list[int]:
    text = str(raw if raw is not None else "0")
    devices: list[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        if not p.isdigit():
            raise ValueError(f"invalid device id {p!r}; expected comma-separated integers")
        devices.append(int(p))
    if not devices:
        raise ValueError("empty device list")
    return devices


def probe_local_arch(backend: str, device_id: int = 0) -> Optional[str]:
    b = (backend or "").strip().lower()
    if b == "ascend":
        return _probe_arch_ascend(device_id)
    if b == "cuda":
        return _probe_arch_cuda(device_id)
    if b == "cpu":
        return normalize_cpu_arch_name(platform.machine())
    return None


def _probe_arch_ascend(device_id: int) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["npu-smi", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    match = re.search(rf"^\|\s*{int(device_id)}\s+(\S+)\s*\|", proc.stdout, re.MULTILINE)
    if not match:
        return None
    return normalize_ascend_arch_name(match.group(1))


def _probe_arch_cuda(device_id: int) -> Optional[str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                "-i",
                str(int(device_id)),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    name = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    return normalize_cuda_arch_name(name) if name else None
