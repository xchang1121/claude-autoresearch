"""Local eval — direct subprocess transport for --devices.

Mirrors the worker server's /run endpoint but in-process: extract the
tar.gz via the shared `safe_extract`, run the generated `eval_<op>.py`
script TWICE (once ref-only, once kernel-only) so a kernel-induced
SIGKILL / device hang in the kernel pass can't take down ref data that
the ref pass already wrote. Merge per-phase sidecars and return the
dict shape both transports share. When a sticky baseline is supplied
the ref pass is skipped.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from typing import Optional

from .eval_runner import (
    build_response, env_for, merge_sidecars, read_sidecar, safe_extract,
    write_merged_sidecar,
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
    """Extract package, run eval_<op>.py in two passes (ref + kernel),
    merge per-phase sidecars, return the worker /run shape:
        {"device_id", "returncode", "log", "eval_result"}.

    Skips the ref pass entirely when override_base_us is set (sticky
    baseline — ref doesn't need re-measuring this round).
    """
    skip_ref = (override_base_us is not None and override_base_us > 0)
    script = f"eval_{op_name}.py"

    with tempfile.TemporaryDirectory(prefix=f"ar_local_{op_name}_") as tmp:
        try:
            safe_extract(package_bytes, tmp)
        except Exception as e:
            return build_response(
                device_id, 1, f"extract failed: {e}", None)

        # --- Pass 1: ref-only subprocess (immune to kernel-side death) -
        if skip_ref:
            ref_log = ""
            ref_payload: Optional[dict] = None
        else:
            ref_env = env_for(
                device_id,
                override_base_us=override_base_us,
                override_base_per_shape_us=override_base_per_shape_us,
                phase="ref_only",
                sidecar_path=os.path.join(tmp, "eval_result_ref.json"),
            )
            _, ref_log = _run_script(tmp, script, ref_env, timeout)
            ref_payload = read_sidecar(tmp, "eval_result_ref.json")

        # --- Pass 2: kernel-only subprocess (verify + profile_gen) ----
        # rc is taken from this pass — downstream readers treat the
        # kernel-side returncode as authoritative for round outcomes.
        kernel_env = env_for(
            device_id,
            override_base_us=override_base_us,
            override_base_per_shape_us=override_base_per_shape_us,
            phase="kernel_only",
            sidecar_path=os.path.join(tmp, "eval_result_kernel.json"),
        )
        rc, kernel_log = _run_script(tmp, script, kernel_env, timeout)
        kernel_payload = read_sidecar(tmp, "eval_result_kernel.json")

        merged = merge_sidecars(ref_payload, kernel_payload)
        write_merged_sidecar(tmp, merged)
        log = "\n".join(s for s in (ref_log, kernel_log) if s).strip()
        return build_response(device_id, rc, log, merged)
