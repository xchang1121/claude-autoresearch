import os
import logging
from typing import Annotated, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import uvicorn

from eval.worker.interface import (
    DEFAULT_EVAL_TIMEOUT_S,
    DEFAULT_GEN_REF_TIMEOUT_S,
)
from eval.worker.local_worker import LocalWorker
from eval.worker.device_pool import DevicePool
from eval.json_safe import sanitize_floats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global worker instance
worker: Optional[LocalWorker] = None

def get_worker_config():
    """Get worker configuration from environment variables."""
    backend = os.environ.get("WORKER_BACKEND", "cuda")
    arch = os.environ.get("WORKER_ARCH", "a100")
    devices_str = os.environ.get("WORKER_DEVICES", "0")

    try:
        devices = [int(d.strip()) for d in devices_str.split(",")]
    except ValueError:
        logger.warning(f"Invalid WORKER_DEVICES: {devices_str}, using [0]")
        devices = [0]

    return backend, arch, devices

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize worker resources on startup."""
    global worker
    backend, arch, devices = get_worker_config()

    logger.info(f"Initializing Worker Service: Backend={backend}, Arch={arch}, Devices={devices}")

    device_pool = DevicePool(devices)
    worker = LocalWorker(device_pool, backend=backend)

    yield

    # Cleanup if needed
    logger.info("Shutting down Worker Service")

app = FastAPI(title="AIKG Worker Service", lifespan=lifespan)

@app.post("/api/v1/verify")
async def verify(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    timeout: int = Form(DEFAULT_EVAL_TIMEOUT_S)
):
    """Execute verification and return success, log, and artifacts."""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    try:
        logger.info(f"[{task_id}] Received verification request for {op_name}")

        # Read package data
        package_data = await package.read()

        # Execute verification (now returns artifacts)
        success, log, artifacts = await worker.verify(package_data, task_id, op_name, timeout)

        return sanitize_floats({
            "success": success,
            "log": log,
            "artifacts": artifacts
        })

    except Exception as e:
        logger.error(f"[{task_id}] Verification request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/profile")
async def profile(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    profile_settings: str = Form("{}")
):
    """
    Execute profiling task.
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    import json
    try:
        settings = json.loads(profile_settings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for profile_settings")

    try:
        package_data = await package.read()
        result = await worker.profile(package_data, task_id, op_name, settings)
        return sanitize_floats(result)
    except Exception as e:
        logger.error(f"[{task_id}] Profiling request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate_reference")
async def generate_reference(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    timeout: int = Form(DEFAULT_GEN_REF_TIMEOUT_S)
):
    """Generate reference data by running the packaged task reference."""
    import base64

    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    try:
        logger.info(f"[{task_id}] Received generate_reference request for {op_name}")

        package_data = await package.read()

        success, log, ref_bytes = await worker.generate_reference(
            package_data, task_id, op_name, timeout
        )

        if success:
            # Return binary reference data as base64.            ref_data_b64 = base64.b64encode(ref_bytes).decode('utf-8')
            return {
                "success": True,
                "log": log,
                "reference_data": ref_data_b64
            }
        else:
            return {
                "success": False,
                "log": log,
                "reference_data": ""
            }

    except Exception as e:
        logger.error(f"[{task_id}] Generate reference request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/profile_single_task")
async def profile_single_task(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    profile_settings: str = Form("{}")
):
    """Measure one task without comparing it against a baseline."""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    import json
    try:
        settings = json.loads(profile_settings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for profile_settings")

    try:
        logger.info(f"[{task_id}] Received profile_single_task request for {op_name}")

        package_data = await package.read()
        result = await worker.profile_single_task(package_data, task_id, op_name, settings)
        return sanitize_floats(result)

    except Exception as e:
        logger.error(f"[{task_id}] Profile single task request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/docs/{doc_name}")
async def get_doc(
    doc_name: str,
):
    """Return a documentation payload available in the worker environment."""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    try:
        content = await worker.get_doc(doc_name)
        return {
            "doc_name": doc_name,
            "content": content,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Get doc request failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/acquire_device")
async def acquire_device(
    task_id: str = Form(...)
):
    """
    Acquire a device from the device pool.
    Client should call this before generating verification scripts.
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    try:
        device_id = await worker.device_pool.acquire_device()
        logger.info(f"[{task_id}] Acquired device {device_id}")
        return {"device_id": device_id}
    except Exception as e:
        logger.error(f"[{task_id}] Failed to acquire device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/release_device")
async def release_device(
    task_id: str = Form(...),
    device_id: int = Form(...)
):
    """
    Release a device back to the device pool.
    Client should call this after task completion.
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    try:
        await worker.device_pool.release_device(device_id)
        logger.info(f"[{task_id}] Released device {device_id}")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"[{task_id}] Failed to release device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status")
async def status():
    """Daemon liveness + identity. ``log_file`` echoes the daemon's stdout
    log path set by worker_service so the remote probe tails the actual file instead of guessing."""
    log_file = os.environ.get("AR_WORKER_LOG_FILE") or os.environ.get("AKG_WORKER_LOG_FILE") or ""
    if worker is None:
        return {"status": "initializing", "log_file": log_file}

    backend, arch, devices = get_worker_config()
    return {
        "status": "ready",
        "backend": backend,
        "arch": arch,
        "devices": devices,
        "log_file": log_file,
    }


@app.get("/api/v1/health")
async def health():
    """Non-blocking daemon health probe.

    /status only checks that HTTP is online. This endpoint briefly exercises
    the same device-pool queue path used by real eval requests without
    occupying a device. A fully busy queue is still healthy.
    """
    import asyncio

    if worker is None:
        return {"status": "initializing", "healthy": False, "free": 0}

    backend, arch, devices = get_worker_config()
    device_pool = worker.device_pool
    pool = device_pool.available_devices
    base = {
        "status": "ready",
        "backend": backend,
        "arch": arch,
        "devices": devices,
        "free": pool.qsize(),
        "healthy": False,
    }

    async def _probe():
        async with device_pool.condition:
            try:
                device_id = pool.get_nowait()
            except asyncio.QueueEmpty:
                return None
            pool.put_nowait(device_id)
            device_pool.condition.notify()
            return device_id

    try:
        device_id = await asyncio.wait_for(_probe(), timeout=5.0)
        base["healthy"] = True
        if device_id is not None:
            base["probed_device"] = device_id
        else:
            base["note"] = "all devices busy (healthy, just at capacity)"
        return base
    except asyncio.TimeoutError:
        base["error"] = "event loop unresponsive (>5s)"
        logger.warning("health probe timed out: event loop did not respond within 5s")
        return base
    except Exception as e:
        base["error"] = f"health probe failed: {type(e).__name__}: {e}"
        logger.warning("health probe failed: %s", e)
        return base


def start_server(host: Optional[str] = None, port: Optional[int] = None):
    """Start the worker HTTP service."""
    if host is None:
        host = os.environ.get("WORKER_HOST", "0.0.0.0")
    if port is None:
        port = int(os.environ.get("WORKER_PORT", "9001"))

    logger.info(f"Starting Worker Service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
