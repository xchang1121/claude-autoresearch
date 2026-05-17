"""Local eval — direct subprocess transport for --devices.

Mirrors the worker server's /run endpoint but in-process: extract the
tar.gz, run the SINGLE auto-generated `eval_<op>.py` script (which does
verify + profile_gen + profile_base in one Python process so JIT and
autotune state stay warm), read `eval_result.json` from the extracted
dir, return the same dict shape the worker does.
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


def _env_for(device_id: int,
             override_base_us: Optional[float] = None) -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DEVICE_ID"] = str(device_id)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows libiomp5 double-load
    if override_base_us is not None and override_base_us > 0:
        env["AR_OVERRIDE_BASE_TIME_US"] = f"{override_base_us:.6f}"
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


def local_eval(package_bytes: bytes, op_name: str, timeout: int,
               device_id: int,
               override_base_us: Optional[float] = None) -> dict:
    """Extract package, run eval_<op>.py, return the worker /run shape:
        {"device_id", "returncode", "log", "eval_result"}.
    `eval_result` is the parsed contents of `eval_result.json`, or None
    if the script crashed before writing it.
    """
    with tempfile.TemporaryDirectory(prefix=f"ar_local_{op_name}_") as tmp:
        try:
            with tarfile.open(fileobj=io.BytesIO(package_bytes), mode="r:gz") as tar:
                tar.extractall(tmp)
        except Exception as e:
            return {
                "device_id": device_id,
                "returncode": 1,
                "log": f"extract failed: {e}",
                "eval_result": None,
            }

        env = _env_for(device_id, override_base_us=override_base_us)
        rc, log = _run_script(tmp, f"eval_{op_name}.py", env, timeout)

        eval_result = None
        sidecar = os.path.join(tmp, "eval_result.json")
        if os.path.isfile(sidecar):
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    eval_result = json.load(f)
            except Exception as e:
                logger.warning("local_eval: failed to parse sidecar: %s", e)

        return {
            "device_id": device_id,
            "returncode": rc,
            "log": log,
            "eval_result": eval_result,
        }
