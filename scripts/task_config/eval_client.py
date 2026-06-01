"""Eval dispatcher — one entry point, two transports.

`run_eval(task_dir, config, device_id=None, worker_urls=None) ->
EvalResult` routes:

  - `worker_urls` (CLI arg or task.yaml `worker.urls`) non-empty → ship
    a tar.gz package via HTTP POST to the first reachable worker.
  - Else probe the Ascend runtime; on success drive `eval_kernel.py` in
    two passes locally (`utils.eval_runner.local_eval`).
  - Else → EvalResult(INFRA_FAIL) explaining the missing transport.

Request-time logic (case-count probe, timeout scaling, sticky lookup)
lives in `eval_request`; response interpretation lives in
`eval_assemble`. Everything else here is helpers — transport detail,
not control flow.
"""
import json
import os
import sys
import uuid
from typing import Optional
from urllib.request import Request, urlopen

from .eval_assemble import assemble_eval_result as _assemble_eval_result
from .eval_request import build_eval_request
from .loader import TaskConfig, py_stem
from .metric_policy import EvalOutcome, EvalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_request(prefix: str, request) -> None:
    if request.num_cases > 1:
        print(f"[{prefix}] eval_timeout scaled per shape: "
              f"{request.config.eval_timeout}s/shape x "
              f"{request.num_cases} cases = {request.timeout}s",
              file=sys.stderr)
    note = request.sticky_note()
    if note:
        print(f"[{prefix}] Skipping ref profile; {note}", file=sys.stderr)


def _normalize_worker_url(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        url = f"http://{url}"
    return url.rstrip("/")


def _worker_status(worker_url: str,
                   timeout: Optional[float] = None) -> Optional[dict]:
    if timeout is None:
        # Lazy import: _select_worker is the only caller and it runs after
        # run_eval puts scripts/ on sys.path.
        _scripts_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from utils.settings import worker_status_timeout
        timeout = worker_status_timeout()
    try:
        req = Request(f"{worker_url}/api/v1/status", method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _select_worker(urls: list) -> Optional[str]:
    """Prefer the reachable worker with the most free device slots; ties
    broken by input order. Returns None when no URL is reachable."""
    best = None  # (-free, idx, url)
    for idx, u in enumerate(urls):
        st = _worker_status(u)
        if st is None:
            continue
        free = st.get("free", 0)
        if not isinstance(free, int):
            free = 0
        cand = (-free, idx, u)
        if best is None or cand < best:
            best = cand
    return best[2] if best is not None else None


def _multipart_post(url: str, fields: dict, files: dict,
                    timeout: float) -> dict:
    """POST multipart/form-data using only stdlib. Returns parsed JSON."""
    boundary = f"----AutoResearch{uuid.uuid4().hex}"
    body = b""
    for name, value in fields.items():
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                 f"{value}\r\n").encode("utf-8")
    for name, (filename, data, content_type) in files.items():
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{name}"; '
                 f'filename="{filename}"\r\n'
                 f"Content-Type: {content_type}\r\n\r\n").encode("utf-8")
        body += data if isinstance(data, bytes) else data.encode("utf-8")
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")

    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _pick_device(config: TaskConfig, device_id: Optional[int]) -> int:
    if device_id is not None:
        return int(device_id)
    if config.devices:
        return int(config.devices[0])
    # No explicit device on the call AND task.yaml has no `devices`
    # field. Fall back to NPU 0 with a loud warning — legitimate
    # callers (notebooks, ad-hoc reruns) do hit this path, but a
    # SILENT fallback to 0 is what once let `--devices 6` get
    # rewritten to 0 and OOM on a busy NPU.
    print(
        "[local_eval] WARNING: no device specified (no device_id arg, "
        "no `devices` field in task.yaml). Defaulting to NPU 0. If "
        "another card is intended, pass --device-id N or set "
        "`devices: [N]` in task.yaml.",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_eval(task_dir: str, config: TaskConfig,
             device_id: Optional[int] = None,
             worker_urls: Optional[list] = None) -> EvalResult:
    """Pick a transport, run eval, assemble EvalResult.

    Routing:
      1. CLI `worker_urls` or `config.worker_urls` non-empty → remote.
      2. Else if Ascend runtime probe succeeds → local subprocess.
      3. Else → INFRA_FAIL.
    """
    urls = [_normalize_worker_url(u)
            for u in (worker_urls or config.worker_urls or []) if u]

    # ----- Remote transport ----------------------------------------------
    if urls:
        worker_url = _select_worker(urls)
        if worker_url is None:
            return EvalResult(
                outcome=EvalOutcome.INFRA_FAIL,
                error=f"no reachable worker from: {urls}",
            )

        request = build_eval_request(task_dir, config)
        task_id = f"{config.name}_{uuid.uuid4().hex[:8]}"
        _log_request("remote_eval", request)
        print(f"[remote_eval] worker={worker_url} task={task_id}",
              file=sys.stderr)

        try:
            from .package_builder import build_package
            package = build_package(task_dir, config)
        except Exception as e:
            return EvalResult(outcome=EvalOutcome.INFRA_FAIL,
                              error=f"failed to build package: {e}")

        fields = {
            "task_id": task_id,
            "op_name": request.config.name,
            "timeout": str(int(request.timeout)),
        }
        if request.override_base_us is not None and request.override_base_us > 0:
            fields["override_base_us"] = f"{request.override_base_us:.6f}"
        if request.override_base_per_shape_us:
            fields["override_base_per_shape_us"] = json.dumps(
                [float(v) for v in request.override_base_per_shape_us])

        try:
            resp = _multipart_post(
                f"{worker_url}/api/v1/run",
                fields=fields,
                files={"package": ("package.tar.gz", package,
                                   "application/gzip")},
                # Worker runs ref(profile_base) + kernel(verify,profile_gen)
                # sequentially, each bounded by request.timeout; only the
                # kernel pass runs when sticky lets it skip base profiling.
                # Wait for the worst-case wall time, else the client
                # disconnects mid-eval and the worker aborts with HTTP 499.
                timeout=request.timeout * (1 if request.sticky else 2) + 60,
            )
        except Exception as e:
            # Pull the response body out of HTTPError so structured worker
            # errors (e.g. drift-guard "restart required, changed=[...]"
            # at 503; "client disconnected while queued" at 499) reach the
            # user instead of being squashed into "HTTP Error 503: Service
            # Unavailable". Without this, the operator only sees the
            # status line and has to ssh into the worker log to find out
            # whether to restart the daemon or chase an NPU fault.
            from urllib.error import HTTPError as _HTTPError
            detail = str(e)
            if isinstance(e, _HTTPError):
                try:
                    body_bytes = e.read()
                    if body_bytes:
                        try:
                            body = json.loads(body_bytes.decode("utf-8",
                                                                errors="replace"))
                            body_str = (body.get("detail")
                                        if isinstance(body, dict)
                                        else None) or json.dumps(body)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            body_str = body_bytes.decode(
                                "utf-8", errors="replace")
                        detail = f"HTTP {e.code}: {body_str[:500]}"
                except Exception:
                    pass
            return EvalResult(outcome=EvalOutcome.INFRA_FAIL,
                              error=f"worker /run failed: {detail}")

        return _assemble_eval_result(
            resp.get("verify_resp") or {},
            resp.get("profile_resp") or {},
        )

    # ----- Local transport -----------------------------------------------
    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from utils.eval_runner import detect_local_backend, local_eval
    ok, why = detect_local_backend()
    if not ok:
        return EvalResult(
            outcome=EvalOutcome.INFRA_FAIL,
            error=(
                f"ascend runtime unavailable: {why}. Install torch + "
                f"torch_npu + CANN locally, or pass --worker-url to "
                f"ship eval to a remote NPU host."
            ),
        )

    dev = _pick_device(config, device_id)
    request = build_eval_request(task_dir, config)
    _log_request("local_eval", request)

    kernel_basename = (py_stem(config.editable_files[0])
                       if config.editable_files else "kernel")
    ref_basename = py_stem(config.ref_file)
    print(f"[local_eval] device={dev}; eval_kernel.py "
          f"(verify + profile_gen"
          f"{'' if request.sticky else ' + profile_base'})...",
          file=sys.stderr)

    verify_resp, profile_resp = local_eval(
        task_dir=task_dir,
        op_name=config.name,
        kernel_file=kernel_basename,
        ref_file=ref_basename,
        timeout=request.timeout,
        device_id=dev,
        override_base_time_us=request.override_base_us,
        override_base_per_shape_us=request.override_base_per_shape_us,
    )
    return _assemble_eval_result(verify_resp, profile_resp)
