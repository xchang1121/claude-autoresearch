"""In-process bridge to the `ar_cli.py env` CLI.

All env / hardware / DSL-table queries route through subprocess to
``python .autoresearch/scripts/ar_cli.py env <subcmd>``. Results are
cached for the lifetime of the process — each query is sub-second but
called many times during arg parsing, scaffolding, and package
assembly.

The function names here mirror what the old ``utils/hw_detect.py``
exposed so call sites (``scaffold.py``, ``task_config/package_builder.py``,
``engine/parse_args.py``) only need their import line touched.

Why subprocess instead of importing the env probes directly: it keeps a
single source of truth for the DSL→backend table and the npu-smi /
nvidia-smi probes — ``ar_cli env`` is the public contract, and any
future change to that contract (new fields, new DSL aliases) becomes
visible to every caller through one process boundary instead of N
import paths drifting independently.
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_RESULT_SENTINEL = "AR_ENV_RESULT:"
_AR_CLI = str(Path(__file__).resolve().parent.parent / "ar_cli.py")


def _run(args: list, timeout: float = 30.0) -> Optional[dict]:
    cmd = [sys.executable, _AR_CLI, "env", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith(_RESULT_SENTINEL):
            try:
                return json.loads(line[len(_RESULT_SENTINEL):])
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------------------
# DSL table
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def list_dsls() -> list:
    """List of `{name, backend, device_type}` entries. Cached per process."""
    data = _run(["list-dsls"]) or {}
    return list(data.get("dsls") or [])


def backend_for_dsl(dsl: str) -> str:
    """Return the backend for `dsl`; raises `ValueError` on unknown DSL."""
    key = (dsl or "").lower()
    for entry in list_dsls():
        if entry.get("name") == key:
            return entry["backend"]
    known = sorted(d.get("name", "") for d in list_dsls())
    raise ValueError(f"Unknown DSL {dsl!r}; known: {known}")


def device_type_for_backend(backend: str) -> str:
    """Return torch.device prefix ('npu' / 'cuda' / 'cpu') for backend."""
    key = (backend or "").lower()
    for entry in list_dsls():
        if entry.get("backend") == key:
            dt = entry.get("device_type")
            if dt:
                return dt
    # Fall back to the well-known map without a CLI round trip — used by
    # _detect_device_type() during package_builder import, where a stale
    # subprocess on a misconfigured system would otherwise block.
    fallback = {"ascend": "npu", "cuda": "cuda", "cpu": "cpu"}.get(key)
    if fallback is None:
        raise ValueError(f"Unknown backend {backend!r}")
    return fallback


# ---------------------------------------------------------------------------
# Local arch derivation
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _detect_local() -> dict:
    return _run(["detect"]) or {}


def derive_arch(backend: str, device_id: int) -> Optional[str]:
    """Look up arch for (backend, device_id) from `ar_cli env detect` output.

    Returns None if backend is unavailable on this host or the device id
    is not present. Callers decide whether None is fatal — scaffold
    treats it as a hard error, but parse_args only uses it for the
    informational `arch` field.
    """
    backends = (_detect_local().get("backends") or {})
    info = backends.get((backend or "").lower())
    if not info or not info.get("available"):
        return None
    for dev in info.get("devices") or []:
        if dev.get("id") == device_id:
            return dev.get("arch")
    # No exact id match — fall back to the backend-level arch (homogeneous host).
    return info.get("arch")


# ---------------------------------------------------------------------------
# Remote worker status
# ---------------------------------------------------------------------------

def fetch_worker_hardware(worker_url: str) -> Optional[dict]:
    """GET `/api/v1/status` on a worker via `ar_cli env detect --worker-url`.

    Returns the worker's status dict (the same shape `worker/server.py`
    emits — keys: `status`, `backend`, `arch`, `devices`, `free`), or
    `None` if the worker is unreachable. The wrapper keys (`remote` /
    `url` / `ok`) added by the CLI are stripped so callers see exactly
    what the worker returned, matching the old hw_detect contract.
    """
    data = _run(["detect", "--worker-url", worker_url], timeout=15.0)
    if not data or not data.get("ok"):
        return None
    out = dict(data)
    for k in ("remote", "url", "ok"):
        out.pop(k, None)
    return out
