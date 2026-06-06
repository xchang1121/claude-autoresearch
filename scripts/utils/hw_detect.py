"""Hardware detection: derive an arch string from a local device id.

The repo target backend is pinned in config.yaml. This module only owns the
small backend-specific probes that scaffold needs before a task is created.
"""
from __future__ import annotations

import platform
import re
import subprocess
from typing import Optional


def derive_arch(device_id: int, backend: str = "ascend") -> Optional[str]:
    """Return the arch string for ``device_id`` on ``backend``.

    Probe per backend:
      - ``ascend``: parse ``npu-smi info``.
      - ``cuda``: parse ``nvidia-smi --query-gpu=name``.
      - ``cpu``: normalize ``platform.machine()``.
    """
    backend = (backend or "ascend").lower()
    if backend == "ascend":
        return _derive_arch_ascend(device_id)
    if backend == "cuda":
        return _derive_arch_cuda(device_id)
    if backend == "cpu":
        return _derive_arch_cpu()
    return None


def probe_hint(backend: str) -> str:
    """Human hint for failed arch probes."""
    backend = (backend or "ascend").lower()
    return {
        "ascend": "is npu-smi on PATH?",
        "cuda": "is nvidia-smi on PATH?",
        "cpu": "platform.machine() returned empty",
    }.get(backend, f"no arch probe registered for backend={backend!r}")


def _derive_arch_ascend(device_id: int) -> Optional[str]:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    pattern = re.compile(rf"^\|\s*{int(device_id)}\s+(\S+)\s*\|", re.MULTILINE)
    match = pattern.search(result.stdout)
    if not match:
        return None
    return f"ascend{match.group(1).strip().lower()}"


_CUDA_MARKETING_NOISE = re.compile(
    r"\b(?:nvidia|tesla|geforce|quadro|titan|laptop|gpu|pcie|sxm\d*|hbm\d?|"
    r"\d+\s*gb)\b",
    re.IGNORECASE,
)
_CUDA_MODEL_PAT = re.compile(
    r"\b(rtx[\s\-]*\d{3,4}[a-z]?|gtx[\s\-]*\d{3,4}[a-z]?|[ahvltb]\d{1,4}[a-z]?)\b",
    re.IGNORECASE,
)


def _derive_arch_cuda(device_id: int) -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                "-i",
                str(int(device_id)),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    name = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not name:
        return None
    cleaned = _CUDA_MARKETING_NOISE.sub(" ", name)
    match = _CUDA_MODEL_PAT.search(cleaned)
    if not match:
        return None
    return re.sub(r"[\s\-]+", "", match.group(1).lower())


def _derive_arch_cpu() -> Optional[str]:
    machine = (platform.machine() or "").lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return None
