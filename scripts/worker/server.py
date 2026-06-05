"""AutoResearch worker — single HTTP endpoint that runs eval locally.

Endpoints:
    GET  /api/v1/status   server-side hardware info (no auth, no body)
    POST /api/v1/run      verify + profile in one round trip

The client ships a tar.gz built by `task_config.package_builder`. We
`safe_extract` it into a tempdir, pick a device slot from an asyncio
queue, and hand the extracted dir to `utils.eval_runner.local_eval` —
the exact code path direct local eval uses. The (verify_resp,
profile_resp) tuple it returns is JSON-serialised back to the client,
which feeds them to `task_config.eval_assemble.assemble_eval_result`.

Device assignment is internal to /run, so clients cannot leak a slot
(no acquire/release endpoints).

Non-finite floats (inf / -inf / nan) from a 0us-latency kernel or a
crashed-profile parse are recursively rewritten to `null` before
serialisation — FastAPI's JSON encoder rejects them with HTTP 500
otherwise.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import tempfile
import tarfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

# worker/server.py → autoresearch/scripts/ is one parent up.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from task_config import load_task_config, py_stem  # noqa: E402
from utils.eval_runner import local_eval_async  # noqa: E402
from utils.json_io import sanitize_floats as _sanitize_floats  # noqa: E402
from utils.settings import (  # noqa: E402
    worker_port as _worker_port,
    default_eval_timeout as _default_eval_timeout,
)

logger = logging.getLogger(__name__)

_state: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_devices(s: str) -> list[int]:
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError:
        logger.warning("WORKER_DEVICES=%r unparseable; defaulting to [0]", s)
        return [0]


def _safe_extract_tar(tar_bytes: bytes, dest_dir: str) -> None:
    """Reject path-traversal entries (members whose resolved path escapes
    `dest_dir`) before unpacking. Refuses symlinks / devices outright."""
    import io
    dest_abs = os.path.realpath(dest_dir)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        for m in tar.getmembers():
            if m.issym() or m.islnk() or m.isdev() or m.isfifo():
                raise ValueError(f"unsafe tar member type: {m.name}")
            target = os.path.realpath(os.path.join(dest_abs, m.name))
            if not (target == dest_abs or target.startswith(dest_abs + os.sep)):
                raise ValueError(f"path traversal blocked: {m.name}")
        tar.extractall(dest_abs)


# ---------------------------------------------------------------------------
# Startup janitor — clean leftover /tmp/ar_run_* dirs
# ---------------------------------------------------------------------------
# `_run_eval_async` uses TemporaryDirectory(prefix="ar_run_…"), which
# auto-cleans on the context manager exit. But the worker process
# itself can be SIGKILLed (OOM-killer, hung NPU, operator -9), and
# then every in-flight eval's tmp dir leaks under /tmp/ until disk
# pressure forces investigation. The packages there include the task
# source the user shipped, so unbounded retention is also a
# data-hygiene problem.
#
# Run on startup: a freshly started worker has zero in-flight evals
# by definition, so any pre-existing /tmp/ar_run_* dir came from a
# previous (crashed) worker and is safe to remove. Best-effort —
# per-dir errors are logged and skipped so a single permissions issue
# doesn't keep the worker from coming up.

def _janitor_clean_stale_tmp_dirs() -> int:
    """Remove every `/tmp/ar_run_*` dir left behind by a previous
    worker that didn't clean up. Returns the count removed."""
    import glob
    import shutil
    tmp_root = tempfile.gettempdir()
    cleaned = 0
    for path in glob.glob(os.path.join(tmp_root, "ar_run_*")):
        if not os.path.isdir(path):
            continue
        try:
            shutil.rmtree(path)
            cleaned += 1
        except OSError as e:
            logger.warning("janitor: could not remove %s: %s", path, e)
    return cleaned


# ---------------------------------------------------------------------------
# Source-drift detection
# ---------------------------------------------------------------------------
# Daemon-style workers keep `utils.eval_runner` (and friends) cached in
# sys.modules across requests. If the operator `git pull`s the source
# tree between requests the process keeps serving the in-memory copy,
# which is how a wire-format-broken commit (e.g. 447da0f, which made
# every fresh /run ModuleNotFoundError) "passed" smoke and batch tests
# in the same session — the worker had been started before the offending
# pull. The guard below snapshots .py mtimes once at startup, then
# rejects /run when any tracked file has been touched since.

def _snapshot_python_files(root: str) -> dict[str, float]:
    """Map of `relpath -> st_mtime` for every .py under `root`. stat
    failures (permission, race during walk) are skipped so a partial
    snapshot is preferred to a hard startup failure."""
    snap: dict[str, float] = {}
    for dp, _dirs, files in os.walk(root):
        # __pycache__ regenerates on .py import; tracking it would
        # produce false drift the first time a fresh worker imports
        # anything not yet compiled.
        if "__pycache__" in dp.split(os.sep):
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            full = os.path.join(dp, f)
            try:
                snap[os.path.relpath(full, root)] = os.path.getmtime(full)
            except OSError:
                continue
    return snap


def _config_yaml_hash() -> Optional[str]:
    """SHA-256 of config.yaml on disk, or None when unreadable. Worker
    startup snapshots this alongside the .py mtime set; without it,
    retunes to eval.warmup / repeats / worker defaults made WHILE the
    worker is up land silently — utils.settings._raw is @lru_cache'd
    per process, so the daemon keeps using the boot-time values
    indefinitely. The drift guard at /run treats config.yaml the same
    way it treats a .py edit: refuse + tell the operator to restart."""
    import hashlib
    cfg = os.path.join(os.path.dirname(_SCRIPTS_DIR), "config.yaml")
    try:
        with open(cfg, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _check_source_drift() -> dict:
    """Compare the live tree against the startup snapshot. Returns
    `{"changed": [...], "added": [...], "removed": [...],
      "config_yaml": "changed" | None}`. Empty lists + config_yaml
    None mean the on-disk tree still matches what's in memory."""
    snap_before: dict[str, float] = _state.get("source_snapshot") or {}
    snap_now = _snapshot_python_files(_SCRIPTS_DIR)
    changed = [p for p, mt in snap_now.items()
               if p in snap_before and snap_before[p] != mt]
    added = [p for p in snap_now if p not in snap_before]
    removed = [p for p in snap_before if p not in snap_now]
    config_hash_before = _state.get("config_hash")
    config_hash_now = _config_yaml_hash()
    # Three drift-worthy transitions; only None→None is clean.
    #   1. hash != hash         — config edited in place
    #   2. hash → None          — config.yaml was readable at boot,
    #                             now isn't (deleted, perms broken).
    #                             Worker would otherwise keep serving
    #                             stale settings while the operator
    #                             thought they were resetting config.
    #   3. None → hash          — opposite: config didn't exist at boot,
    #                             exists now. utils.settings._raw lru-
    #                             caches the absent-file result, so
    #                             without this signal the worker would
    #                             silently keep zero-config defaults.
    if config_hash_before == config_hash_now:
        config_drift = None
    else:
        config_drift = "changed"
    return {
        "changed": sorted(changed)[:10],
        "added":   sorted(added)[:10],
        "removed": sorted(removed)[:10],
        "config_yaml": config_drift,
    }


def _drift_is_clean(drift: dict) -> bool:
    return not (drift["changed"] or drift["added"]
                or drift["removed"] or drift.get("config_yaml"))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = os.environ.get("WORKER_BACKEND", "ascend")
    arch = os.environ.get("WORKER_ARCH", "")
    devices = _parse_devices(os.environ.get("WORKER_DEVICES", "0"))
    q: asyncio.Queue = asyncio.Queue()
    for d in devices:
        q.put_nowait(d)
    snap = _snapshot_python_files(_SCRIPTS_DIR)
    cfg_hash = _config_yaml_hash()
    cleaned = _janitor_clean_stale_tmp_dirs()
    _state.update(backend=backend, arch=arch, devices=devices, queue=q,
                  source_snapshot=snap, config_hash=cfg_hash)
    logger.info("worker ready: backend=%s arch=%s devices=%s "
                "source_snapshot=%d files under %s "
                "config_yaml_hash=%s "
                "janitor_cleaned=%d stale tmp dirs",
                backend, arch, devices, len(snap), _SCRIPTS_DIR,
                (cfg_hash[:12] + "...") if cfg_hash else "<unreadable>",
                cleaned)
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
    # `status` stays "ready" so older clients that key off it (smoke
    # tests, ar_cli pre-eval reachability check) don't suddenly see
    # an unknown state — drift is advisory in /status and only
    # enforced in /run, where the request would otherwise burn an
    # eval slot serving stale code.
    drift = _check_source_drift()
    payload = {
        "status": "ready",
        "backend": _state["backend"],
        "arch": _state["arch"],
        "devices": _state["devices"],
        "free": _state["queue"].qsize(),
    }
    # Only include `code_drift` when something actually drifted —
    # keeps the response identical to the pre-change shape in the
    # common case so JSON-strict consumers don't have to learn a
    # new optional field unless they care.
    if not _drift_is_clean(drift):
        payload["code_drift"] = drift
    return payload


@app.get("/api/v1/health")
async def health():
    """非阻塞健康探活 —— 验"daemon 接 /run 时的请求路径还活着"，但
    不抢占设备：

      - 用 ``asyncio.Queue.get_nowait()`` 试取一次 device，能取就立刻
        放回；空队列（满载）当作 healthy（"忙不是坏"），不报 degraded
      - 整个 handler 5s 超时；超时仅当事件循环本身卡了

    /status 只验证 HTTP server 在线；/health 走一遍真实的 queue 操作
    路径，能抓出"event loop 卡住"或"queue 锁竞争"那类故障。**不会**
    阻塞等设备，所以满载 worker 不会被误判 degraded。"""
    if "queue" not in _state:
        return {"status": "initializing", "healthy": False, "free": 0}

    queue = _state["queue"]
    base = {
        "status": "ready",
        "backend": _state["backend"],
        "arch": _state["arch"],
        "devices": _state["devices"],
        "free": queue.qsize(),
        "healthy": False,
    }

    async def _probe():
        try:
            device_id = queue.get_nowait()
        except asyncio.QueueEmpty:
            # All devices busy — daemon is fine, just at capacity.
            return None
        queue.put_nowait(device_id)
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
        base["error"] = "event loop unresponsive (>5s) —— 事件循环可能卡死"
        logger.warning("健康探活超时：event loop 5 秒内未响应")
        return base
    except Exception as e:
        base["error"] = f"健康探活异常：{type(e).__name__}: {e}"
        logger.warning(f"健康探活异常：{e}")
        return base


@app.post("/api/v1/run")
async def run(
    request: Request,
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    timeout: int = Form(_default_eval_timeout()),
    override_base_us: Optional[float] = Form(None),
    override_base_per_shape_us: Optional[str] = Form(None),
):
    """Verify + profile in one call.

    Device is picked server-side from the queue, used as the DEVICE_ID
    for the eval subprocess, and released in the finally block — clients
    cannot leak a slot.

    The eval runs as an asyncio task; concurrently we watch
    `request.is_disconnected()`. Whichever finishes first wins:

      - eval done   → cancel the watcher, return result
      - client gone → cancel the eval (cascades into a SIGTERM on the
                      eval subprocess group via
                      `utils.eval_runner._run_subprocess_async`), release
                      the device, and return HTTP 499. This is what
                      keeps a `claude --print` killed by its wall-clock
                      cap from leaving an orphan eval pinning the device
                      until the eval finishes naturally.
    """
    if "queue" not in _state:
        raise HTTPException(status_code=503, detail="worker not initialised")

    # Stale-code refuse. If any .py under scripts/ has been touched
    # since startup, the in-memory module set no longer matches what's
    # on disk; serving /run from here would silently use the old code
    # (the exact failure mode that hid 447da0f's broken import for an
    # entire session). Fail loud and tell the operator to restart.
    drift = _check_source_drift()
    if not _drift_is_clean(drift):
        raise HTTPException(
            status_code=503,
            detail=(f"worker code is stale; restart required. "
                    f"changed={drift['changed']}, added={drift['added']}, "
                    f"removed={drift['removed']}, "
                    f"config_yaml={drift.get('config_yaml')}"))

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

    # Stage 1: acquire a device — but race against disconnect so a
    # client that abandons the request while queued doesn't get a
    # device assigned to it the moment one frees up (the previous
    # behaviour acquired silently, then the eval phase's watch_task
    # immediately cancelled, releasing the device — correct but a
    # wasted scheduling window, and the dead client stayed FIFO-
    # blocking until the queue's turn came around).
    queue_task = asyncio.create_task(_state["queue"].get())
    watch_task = asyncio.create_task(_watch_disconnect(request))
    done, _pending = await asyncio.wait(
        [queue_task, watch_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    # Always tear down the queueing-phase watch_task; we either don't
    # need it any more (acquired a device) or it already fired.
    watch_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await watch_task

    if queue_task in done and not queue_task.cancelled():
        device_id = queue_task.result()
    else:
        # Client disconnected before a device freed up. Cancel the
        # pending queue.get; if it raced us and managed to take a
        # device on the very tick we cancelled, return it to the pool
        # so the next caller isn't blocked.
        queue_task.cancel()
        try:
            result = await queue_task
            if isinstance(result, int):
                _state["queue"].put_nowait(result)
        except (asyncio.CancelledError, Exception):
            pass
        logger.info("[%s] client gone while queued; aborting before "
                    "acquiring a device", task_id)
        raise HTTPException(status_code=499,
                            detail="client disconnected while queued")

    try:
        logger.info("[%s] run %s on device %d", task_id, op_name, device_id)
        # Stage 2: run the eval; re-arm watch_task to cancel the eval
        # if the client disconnects mid-flight.
        eval_task = asyncio.create_task(
            _run_eval_async(package_bytes, task_id, op_name, timeout,
                            device_id, override_base_us, per_shape))
        watch_task = asyncio.create_task(_watch_disconnect(request))
        done, _pending = await asyncio.wait(
            [eval_task, watch_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if eval_task in done and not eval_task.cancelled():
            watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watch_task
            try:
                return _sanitize_floats(eval_task.result())
            except Exception as e:
                # Internal eval failure (malformed task.yaml, loader
                # ValueError, response-assembly error). Package it as a
                # structured error response instead of letting it bubble to
                # FastAPI as an opaque HTTP 500 the client can't interpret.
                logger.exception("[%s] eval task raised", task_id)
                return _sanitize_floats(
                    _error_response(device_id, f"worker eval crashed: {e}"))
        # Disconnect (or eval crashed early without a result).
        logger.info("[%s] client gone before eval finished; cancelling, "
                    "releasing device %d", task_id, device_id)
        eval_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await eval_task
        # 499 is the conventional non-RFC "client closed request" code
        # nginx popularised; FastAPI happily relays it.
        raise HTTPException(status_code=499, detail="client disconnected")
    finally:
        await _state["queue"].put(device_id)


async def _watch_disconnect(request: Request) -> None:
    """Block until starlette reports the HTTP client has disconnected.

    `request.is_disconnected()` reads from the starlette receive queue —
    no network traffic, no curl-style heartbeat. uvicorn puts a
    `{"type": "http.disconnect"}` message in that queue the moment the
    TCP socket closes; this coroutine just observes it.

    uvicorn 0.42.0 can report a spurious disconnect immediately after
    the multipart body is consumed (the ASGI receive queue returns
    http.disconnect once form parsing has drained the request stream,
    even though the TCP connection is still alive). A brief initial
    delay lets the transport settle before we start watching — if the
    client is truly gone it will still be gone 3 s later.
    """
    await asyncio.sleep(3)
    while not await request.is_disconnected():
        await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

async def _run_eval_async(package_bytes: bytes, task_id: str, op_name: str,
                          timeout: int, device_id: int,
                          override_base_us: Optional[float],
                          override_base_per_shape_us: Optional[list]
                          ) -> dict:
    """Extract the package, look up kernel/ref filenames from task.yaml,
    dispatch to `local_eval_async`. Returns the dict the client expects.

    Mirrors the sync `_run_eval_sync` this replaced; the difference is
    the eval is `await`-ed (so cancellation propagates into the eval
    subprocess group) instead of run via `asyncio.to_thread` (which
    can't be cancelled — `asyncio.to_thread` runs in a real OS thread
    that doesn't respond to task cancellation, so a SIGTERM'd `claude
    --print` would close its socket and the worker would carry on
    blocking on `subprocess.run` for the rest of the eval).
    """
    with tempfile.TemporaryDirectory(prefix=f"ar_run_{task_id}_") as tmp:
        try:
            _safe_extract_tar(package_bytes, tmp)
        except Exception as e:
            return _error_response(device_id, f"extract failed: {e}")

        config = load_task_config(tmp)
        if config is None:
            return _error_response(device_id,
                                   "task.yaml missing in package")

        kernel_file = (py_stem(config.editable_files[0])
                       if config.editable_files else "kernel")
        ref_file = py_stem(config.ref_file)

        verify_resp, profile_resp = await local_eval_async(
            task_dir=tmp,
            op_name=op_name,
            kernel_file=kernel_file,
            ref_file=ref_file,
            timeout=timeout,
            device_id=device_id,
            override_base_time_us=override_base_us,
            override_base_per_shape_us=override_base_per_shape_us,
        )
        return {
            "device_id": device_id,
            "verify_resp": verify_resp,
            "profile_resp": profile_resp,
        }


def _error_response(device_id: int, msg: str) -> dict:
    # error_source="infra": every caller here is a worker-side
    # infrastructure failure (tar extract, missing task.yaml, internal
    # eval crash) — NOT a kernel defect. assemble_eval_result maps this to
    # INFRA_FAIL so the round isn't charged to the kernel as KERNEL_FAIL.
    return {
        "device_id": device_id,
        "verify_resp": {
            "success": False, "log": msg, "returncode": 1,
            "error_source": "infra", "verify_block": {}, "artifacts": {},
        },
        "profile_resp": {
            "success": False, "log": "", "artifacts": {},
            "gen_time": None, "base_time": None,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def start_server(host: Optional[str] = None, port: Optional[int] = None):
    # SSH-only by design: bind loopback so the worker is reachable ONLY via
    # an ssh -L tunnel (which forwards to the remote's 127.0.0.1). Never
    # bind a public interface — that would expose the eval endpoint on
    # every network.
    host = host or os.environ.get("WORKER_HOST", "127.0.0.1")
    port = int(port or os.environ.get("WORKER_PORT", str(_worker_port())))
    logger.info("starting worker on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    start_server()
