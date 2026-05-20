"""AutoResearch worker — single HTTP endpoint, runs eval script locally.

Endpoints:
    GET  /api/v1/status   server-side hardware info (no auth, no body)
    POST /api/v1/run      verify + profile in one round trip

The client ships a tar.gz built by package_builder. We `safe_extract`
it (rejecting path-traversal), pick a device slot from an internal
asyncio.Queue, run the generated `eval_<op>.py` TWICE (ref pass then
kernel pass) so a kernel-induced SIGKILL / device hang in the kernel
pass can't erase ref data the ref pass already wrote. Per-phase
sidecars are merged into eval_result.json and returned with the
combined stdout+stderr. When a sticky baseline is supplied the ref
pass is skipped.

No acquire/release endpoints, no DevicePool — device assignment is
internal to /run, so the client can't bake a wrong device id into the
package and the pool never decrements for slots that were never
reserved. Extract / env / sidecar / merge helpers are imported from
`utils.eval_runner` so this transport can't drift from the local one.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import uvicorn

# ar_vendored/worker/server.py → .autoresearch/scripts/ is two parents up.
# Insert that on sys.path so we can import the shared runner helpers; the
# worker may launch from any cwd (foreground, daemon, tmux).
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from utils.eval_runner import (  # noqa: E402
    build_response, env_for, merge_sidecars, num_cases_from_kernel_payload,
    read_sidecar, safe_extract, synth_sticky_ref_payload, write_merged_sidecar,
)

logger = logging.getLogger(__name__)

_state: dict = {}


def _parse_devices(s: str) -> list[int]:
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError:
        logger.warning("WORKER_DEVICES=%r unparseable; defaulting to [0]", s)
        return [0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = os.environ.get("WORKER_BACKEND", "cuda")
    arch = os.environ.get("WORKER_ARCH", "")
    devices = _parse_devices(os.environ.get("WORKER_DEVICES", "0"))
    q: asyncio.Queue = asyncio.Queue()
    for d in devices:
        q.put_nowait(d)
    _state.update(backend=backend, arch=arch, devices=devices, queue=q)
    logger.info("worker ready: backend=%s arch=%s devices=%s",
                backend, arch, devices)
    yield
    logger.info("worker shutting down")


app = FastAPI(title="AutoResearch Worker", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/status")
async def status():
    if "queue" not in _state:
        return {"status": "initializing"}
    return {
        "status": "ready",
        "backend": _state["backend"],
        "arch": _state["arch"],
        "devices": _state["devices"],
        "free": _state["queue"].qsize(),
    }


@app.post("/api/v1/run")
async def run(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    timeout: int = Form(600),
    override_base_us: Optional[float] = Form(None),
    override_base_per_shape_us: Optional[str] = Form(None),
):
    """Verify + profile in one call. Device is picked server-side from the
    pool, exported as DEVICE_ID env to the spawned script, and released
    in the finally block — clients cannot leak a slot.

    `override_base_us`: sticky aggregate baseline. When the caller has
    a measured PyTorch reference time, pass it here and the generated
    eval script skips profile_base — saves one full per-shape benchmark
    pass.

    `override_base_per_shape_us`: JSON-encoded list of per-case
    aggregate timings the SEED round measured. When supplied alongside
    override_base_us, the generated script materialises a per_shape
    base profile so speedup_vs_ref stays a geomean of per-shape ratios
    (matching the SEED round's aggregation).
    """
    if "queue" not in _state:
        raise HTTPException(status_code=503, detail="worker not initialised")
    package_bytes = await package.read()
    per_shape: Optional[list] = None
    if override_base_per_shape_us:
        try:
            per_shape = json.loads(override_base_per_shape_us)
            if not (isinstance(per_shape, list) and per_shape):
                per_shape = None
        except Exception as e:
            logger.warning("[%s] bad override_base_per_shape_us JSON: %s",
                           task_id, e)
            per_shape = None
    device_id = await _state["queue"].get()
    try:
        logger.info("[%s] run %s on device %d", task_id, op_name, device_id)
        return await _run_eval(package_bytes, task_id, op_name, timeout,
                               device_id, override_base_us=override_base_us,
                               override_base_per_shape_us=per_shape)
    finally:
        await _state["queue"].put(device_id)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

async def _run_eval(package_bytes: bytes, task_id: str, op_name: str,
                    timeout: int, device_id: int,
                    override_base_us: Optional[float] = None,
                    override_base_per_shape_us: Optional[list] = None) -> dict:
    import tempfile
    skip_ref = (override_base_us is not None and override_base_us > 0)
    script = f"eval_{op_name}.py"

    with tempfile.TemporaryDirectory(prefix=f"ar_run_{task_id}_") as tmp:
        try:
            safe_extract(package_bytes, tmp)
        except Exception as e:
            return build_response(
                device_id, 1, f"extract failed: {e}", None)

        # --- Pass 1: ref-only subprocess (immune to kernel-side death) -
        # Skipped on sticky-baseline rounds; ref_payload is synthesized
        # AFTER pass 2 finishes so we can read num_cases off it.
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
            _, ref_log = await _run_script(tmp, script, ref_env, timeout)
            ref_payload = read_sidecar(tmp, "eval_result_ref.json")

        # --- Pass 2: kernel-only subprocess (verify + profile_gen) ----
        # Kernel-pass rc is the authoritative round outcome.
        kernel_env = env_for(
            device_id,
            override_base_us=override_base_us,
            override_base_per_shape_us=override_base_per_shape_us,
            phase="kernel_only",
            sidecar_path=os.path.join(tmp, "eval_result_kernel.json"),
        )
        rc, kernel_log = await _run_script(tmp, script, kernel_env, timeout)
        kernel_payload = read_sidecar(tmp, "eval_result_kernel.json")

        # Sticky-baseline path: kernel-only pass writes no profile_base
        # (Phase E gated by DO_REF_PHASE=False). Without a synthesized
        # ref payload, merge_sidecars would leave profile_base=None and
        # the round would lose ref_latency_us / speedup_vs_ref entirely.
        if skip_ref:
            ref_payload = synth_sticky_ref_payload(
                override_base_us,
                override_base_per_shape_us,
                num_cases_from_kernel_payload(kernel_payload),
            )

        merged = merge_sidecars(ref_payload, kernel_payload)
        write_merged_sidecar(tmp, merged)
        log = "\n".join(s for s in (ref_log, kernel_log) if s).strip()
        return build_response(device_id, rc, log, merged)


async def _run_script(workdir: str, script: str, env: dict,
                       timeout: int) -> tuple[int, str]:
    """Spawn `python <script>` in workdir. Returns (rc, combined_log).
    rc=124 on timeout (GNU `timeout(1)` convention)."""
    if not os.path.isfile(os.path.join(workdir, script)):
        return 1, f"[worker] missing {script}"

    preexec = os.setsid if hasattr(os, "setsid") else None
    proc = await asyncio.create_subprocess_exec(
        sys.executable, script,
        cwd=workdir, env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=preexec,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout)
        rc = proc.returncode or 0
    except asyncio.TimeoutError:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                await asyncio.sleep(1)
                if proc.returncode is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
        return 124, f"[worker] {script} timed out after {timeout}s"

    log = stdout.decode(errors="replace")
    if stderr:
        log += ("\n" if log else "") + stderr.decode(errors="replace")
    return rc, log.strip()


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def start_server(host: Optional[str] = None, port: Optional[int] = None):
    host = host or os.environ.get("WORKER_HOST", "0.0.0.0")
    port = int(port or os.environ.get("WORKER_PORT", "9001"))
    logger.info("starting worker on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    start_server()
