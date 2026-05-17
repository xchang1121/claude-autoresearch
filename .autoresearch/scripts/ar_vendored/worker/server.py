"""AutoResearch worker — single HTTP endpoint, runs eval scripts locally.

Endpoints:
    GET  /api/v1/status   server-side hardware info (no auth, no body)
    POST /api/v1/run      verify + profile in one round trip

The client (eval_client.py) ships a tar.gz built by package_builder. We
extract it, pick a device slot from an internal asyncio.Queue, run
verify_<op>.py then profile_<op>_base.py + profile_<op>_generation.py as
subprocesses (DEVICE_ID exported via env), and return a single dict the
client converges into an EvalResult.

No acquire/release endpoints, no DevicePool class — device assignment is
internal to /run, which means the client can't bake a wrong device id
into the package and the pool never decrements for slots that were never
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
):
    """Verify + profile in one call. Device is picked server-side from the
    pool, exported as DEVICE_ID env to the spawned scripts, and released
    in the finally block — clients cannot leak a slot."""
    if "queue" not in _state:
        raise HTTPException(status_code=503, detail="worker not initialised")
    package_bytes = await package.read()
    device_id = await _state["queue"].get()
    try:
        logger.info("[%s] run %s on device %d", task_id, op_name, device_id)
        return await _run_eval(package_bytes, task_id, op_name, timeout, device_id)
    finally:
        await _state["queue"].put(device_id)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

async def _run_eval(package_bytes: bytes, task_id: str, op_name: str,
                    timeout: int, device_id: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"ar_run_{task_id}_") as tmp:
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

        verify_rc, verify_log = await _run_script(
            tmp, f"verify_{op_name}.py", env, timeout)
        # Even if verify failed, run both profile scripts: the reference
        # baseline is independent of kernel correctness and is the anchor
        # for speedup. The kernel profile is allowed to fail too.
        _, base_log = await _run_script(
            tmp, f"profile_{op_name}_base.py", env, timeout)
        _, gen_log = await _run_script(
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


def _collect_artifacts(directory: str) -> dict[str, str]:
    """Return {relpath: file_text} for every .json/.jsonl under directory."""
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
