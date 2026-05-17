"""Local eval — direct subprocess transport for --devices.

Mirrors the worker server's /run endpoint but in-process: extract the
tar.gz from package_builder, run verify_<op>.py + profile_<op>_{base,
generation}.py with DEVICE_ID exported, collect JSON artifacts, return
a dict shaped exactly like the worker /run response so the eval_client
result assembler is transport-agnostic.

No msprof/nsys CLI wrapping, no roofline. The DSL adapter's
benchmark_impl (profiler_npu for Ascend, do_bench for CUDA/Triton) is
what gives accurate per-shape timing — it's embedded in the generated
profile script, runs inside the same subprocess, and produces the
*_profile_result.json files this module reads back.
"""
from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def _env_for(device_id: int) -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DEVICE_ID"] = str(device_id)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows libiomp5 double-load
    return env


def _run_script(workdir: str, script: str, env: dict,
                timeout: int) -> tuple[int, str]:
    if not os.path.isfile(os.path.join(workdir, script)):
        return 1, f"[local_eval] missing {script}"

    popen_kwargs: dict = {
        "cwd": workdir, "env": env,
        "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
    }
    if hasattr(os, "setsid"):
        popen_kwargs["preexec_fn"] = os.setsid

    try:
        proc = subprocess.Popen([sys.executable, script], **popen_kwargs)
    except Exception as e:
        return 1, f"failed to launch {script}: {e}"

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode or 0
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            stdout, stderr = b"", b""
        return 124, (stdout or b"").decode(errors="replace") + \
            "\n" + (stderr or b"").decode(errors="replace") + \
            f"\n[local_eval] {script} timed out after {timeout}s"

    log = (stdout or b"").decode(errors="replace")
    if stderr:
        log += ("\n" if log else "") + stderr.decode(errors="replace")
    return rc, log.strip()


def _collect_artifacts(directory: str) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if not (fname.endswith(".json") or fname.endswith(".jsonl")):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, directory).replace("\\", "/")
            try:
                with open(full, "r", encoding="utf-8") as f:
                    artifacts[rel] = f.read()
            except Exception as e:
                logger.warning("cannot read artifact %s: %s", rel, e)
    return artifacts


def _avg_us(artifacts: dict[str, str], key: str) -> Optional[float]:
    raw = artifacts.get(key)
    if not raw:
        return None
    try:
        v = json.loads(raw).get("avg_time_us")
    except Exception:
        return None
    if isinstance(v, (int, float)) and 0 < v < float("inf"):
        return float(v)
    return None


def local_eval(package_bytes: bytes, op_name: str, timeout: int,
               device_id: int) -> dict:
    """Run verify + profile (base + generation) and return the same dict
    shape as the worker /run endpoint."""
    with tempfile.TemporaryDirectory(prefix=f"ar_local_{op_name}_") as tmp:
        try:
            with tarfile.open(fileobj=io.BytesIO(package_bytes), mode="r:gz") as tar:
                tar.extractall(tmp)
        except Exception as e:
            return {
                "device_id": device_id,
                "verify": {"success": False, "log": f"extract failed: {e}",
                            "artifacts": {}},
                "profile": {"log": "", "artifacts": {},
                            "gen_time": None, "base_time": None},
            }

        env = _env_for(device_id)
        verify_rc, verify_log = _run_script(
            tmp, f"verify_{op_name}.py", env, timeout)
        # Both profile scripts always run — kernel failures still let us
        # capture the ref baseline (the user-facing speedup anchor).
        _, base_log = _run_script(
            tmp, f"profile_{op_name}_base.py", env, timeout)
        _, gen_log = _run_script(
            tmp, f"profile_{op_name}_generation.py", env, timeout)

        artifacts = _collect_artifacts(tmp)
        return {
            "device_id": device_id,
            "verify": {
                "success": verify_rc == 0,
                "log": verify_log,
                "artifacts": artifacts,
            },
            "profile": {
                "log": (base_log + "\n" + gen_log).strip(),
                "artifacts": artifacts,
                "gen_time": _avg_us(artifacts, "generation_profile_result.json"),
                "base_time": _avg_us(artifacts, "base_profile_result.json"),
            },
        }
