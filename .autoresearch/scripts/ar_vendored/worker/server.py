"""AutoResearch worker — single HTTP endpoint, runs eval script locally.

Endpoints:
    GET  /api/v1/status   server-side hardware info (no auth, no body)
    POST /api/v1/run      verify + profile in one round trip

The client ships a tar.gz built by package_builder. We extract it,
pick a device slot from an internal asyncio.Queue, run the SINGLE
auto-generated `eval_<op>.py` as one subprocess (verify + profile_gen +
profile_base run in-process there so the JIT/autotune cache stays
warm), then read `eval_result.json` from the extracted dir and return
it together with stdout+stderr to the client.

No acquire/release endpoints, no DevicePool class — device assignment
is internal to /run, so the client can't bake a wrong device id into
the package and the pool never decrements for slots that were never
reserved.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import signal
import sys
import tarfile
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import uvicorn

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
):
    """Verify + profile in one call. Device is picked server-side from the
    pool, exported as DEVICE_ID env to the spawned script, and released
    in the finally block — clients cannot leak a slot.

    `override_base_us`: sticky baseline. When the caller already has a
    measured PyTorch reference time, pass it here and the generated
    eval script skips profile_base — saves one full per-shape benchmark
    pass per round.
    """
    if "queue" not in _state:
        raise HTTPException(status_code=503, detail="worker not initialised")
    package_bytes = await package.read()
    device_id = await _state["queue"].get()
    try:
        logger.info("[%s] run %s on device %d", task_id, op_name, device_id)
        return await _run_eval(package_bytes, task_id, op_name, timeout,
                               device_id, override_base_us=override_base_us)
    finally:
        await _state["queue"].put(device_id)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

async def _run_eval(package_bytes: bytes, task_id: str, op_name: str,
                    timeout: int, device_id: int,
                    override_base_us: Optional[float] = None) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"ar_run_{task_id}_") as tmp:
        try:
            with tarfile.open(fileobj=io.BytesIO(package_bytes), mode="r:gz") as tar:
                tar.extractall(tmp)
        except Exception as e:
            return {
                "device_id": device_id,
                "log": f"extract failed: {e}",
                "eval_result": None,
            }

        env = _env_for(device_id)
        if override_base_us is not None and override_base_us > 0:
            env["AR_OVERRIDE_BASE_TIME_US"] = f"{override_base_us:.6f}"

        rc, log = await _run_script(tmp, f"eval_{op_name}.py", env, timeout)

        eval_result = None
        sidecar = os.path.join(tmp, "eval_result.json")
        if os.path.isfile(sidecar):
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    eval_result = json.load(f)
            except Exception as e:
                logger.warning("[%s] failed to parse sidecar: %s", task_id, e)

        return {
            "device_id": device_id,
            "returncode": rc,
            "log": log,
            "eval_result": eval_result,
        }


def _env_for(device_id: int) -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DEVICE_ID"] = str(device_id)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows libiomp5 double-load
    return env


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
