"""Local eval — direct subprocess transport for --devices.

Mirrors the worker server's /run endpoint but in-process: extract the
tar.gz via the shared `safe_extract`, run the SINGLE generated
`eval_<op>.py` script (verify + profile_gen + profile_base in one
Python process so JIT and autotune state stay warm), read
`eval_result.json` from the extracted dir, return the dict shape both
transports share.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from typing import Optional

from .eval_runner import (
    build_response, env_for, read_sidecar, safe_extract,
)

logger = logging.getLogger(__name__)


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
               override_base_us: Optional[float] = None,
               override_base_per_shape_us: Optional[list] = None) -> dict:
    """Extract package, run eval_<op>.py, return the worker /run shape:
        {"device_id", "returncode", "log", "eval_result"}.
    """
    with tempfile.TemporaryDirectory(prefix=f"ar_local_{op_name}_") as tmp:
        try:
            safe_extract(package_bytes, tmp)
        except Exception as e:
            return build_response(
                device_id, 1, f"extract failed: {e}", None)

        env = env_for(device_id, override_base_us=override_base_us,
                      override_base_per_shape_us=override_base_per_shape_us)
        rc, log = _run_script(tmp, f"eval_{op_name}.py", env, timeout)
        return build_response(device_id, rc, log, read_sidecar(tmp))
